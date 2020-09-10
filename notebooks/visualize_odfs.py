import dipy.core.gradients as dpg
import nibabel as nib
import numpy as np
import os
import s3fs
import shutil

from uuid import uuid4
from dipy.data import get_fnames, default_sphere
from dipy.reconst import csdeconv as csd
from dipy.viz import window, actor


def visualize_odfs_from_files(
    bval_file, bvec_file, dwi_file, sh_coeff_file, sh_order=None
):
    gtab = dpg.gradient_table(bval_file, bvec_file)

    if sh_order is None:
        ndata = np.sum(~gtab.b0s_mask)
        # See dipy.reconst.shm.calculate_max_order
        L1 = (-3 + np.sqrt(1 + 8 * ndata)) / 2.0
        sh_order = int(L1)
        if np.mod(sh_order, 2) != 0:
            sh_order = sh_order - 1
        if sh_order > 8:
            sh_order = 8

    img = nib.load(dwi_file)
    data = img.get_fdata()
    response = csd.auto_response(gtab, data, roi_radius=10, fa_thr=0.7)
    csd_model = csd.ConstrainedSphericalDeconvModel(
        gtab, response[0], sh_order=sh_order
    )
    sh_coeff_img = nib.load(sh_coeff_file)
    sh_coeff = sh_coeff_img.get_fdata()
    csd_fit = csd.SphHarmFit(csd_model, sh_coeff, mask=np.ones(data.shape[:3]))
    csd_odf = csd_fit.odf(default_sphere)

    ren = window.Scene()

    odf_sl = csd_odf[:, :, csd_odf.shape[2] // 2 : csd_odf.shape[2] // 2 + 1, :]

    fodf_spheres = actor.odf_slicer(
        odf_sl, sphere=default_sphere, scale=0.9, norm=False, colormap="viridis"
    )

    ren.add(fodf_spheres)
    window.show(ren)


# TODO: Remember the sh_coeff file looks like './csd_sh_coeff.nii.gz'
def visualize_odfs(
    bval_file, bvec_file, dwi_file, sh_coeff_file, sh_order=None, clobber_dir=False
):
    is_s3 = {
        fname: fname.startswith("s3://")
        for fname in [bval_file, bvec_file, dwi_file, sh_coeff_file]
    }

    local_dir = None

    if any(is_s3.values()):
        local_dir = "odf_viz_" + str(uuid4())
        local_dir = os.path.abspath(os.path.join(os.getcwd(), local_dir))
        os.mkdir(local_dir)

    fs = s3fs.S3FileSystem()

    def download_from_s3(s3_uri):
        s3_key = s3_uri.replace("s3://", "")
        basename = os.path.basename(s3_key)
        local_file = os.path.join(local_dir, basename)
        fs.get(s3_key, local_file)
        return local_file

    file_map = {
        fname: download_from_s3(fname) if fname.startswith("s3://") else fname
        for fname in is_s3.keys()
    }

    visualize_odfs_from_files(
        bval_file=file_map[bval_file],
        bvec_file=file_map[bvec_file],
        dwi_file=file_map[dwi_file],
        sh_coeff_file=file_map[sh_coeff_file],
        sh_order=sh_order,
    )

    if local_dir is not None and clobber_dir:
        shutil.rmtree(local_dir)
