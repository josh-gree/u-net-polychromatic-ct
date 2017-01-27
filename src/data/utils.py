import odl

from PIL import Image, ImageDraw
from odl_fbp import fbp_op

import numpy as np

Nx, Ny = 256, 256
Np_lim, Nd_lim = 20, 50
Np_full, Nd_full = 1000, 500

reco_space = odl.uniform_discr([-1, -1], [1, 1], [Nx, Ny], dtype='float32')

angle_partition_lim = odl.uniform_partition(0, np.pi, Np_lim)
detector_partition_lim = odl.uniform_partition(-1.2, 1.2, Nd_lim)
geometry_lim = odl.tomo.Parallel2dGeometry(angle_partition_lim, detector_partition_lim)

angle_partition_full = odl.uniform_partition(0, np.pi, Np_full)
detector_partition_full = odl.uniform_partition(-1.2, 1.2, Nd_full)
geometry_full = odl.tomo.Parallel2dGeometry(angle_partition_full, detector_partition_full)

ray_trafo_lim = odl.tomo.RayTransform(reco_space, geometry_lim, impl="astra_cuda")
fbp_lim = fbp_op(ray_trafo_lim)

ray_trafo_full = odl.tomo.RayTransform(reco_space, geometry_full, impl="astra_cuda")
fbp_full = fbp_op(ray_trafo_full)

def rand_circs():
    N = np.random.randint(2, 11)
    rs = np.random.uniform(3, 19, size=N)
    ps = np.random.uniform(58.4, 197.6, size=(2, N))
    return rs, ps

def make_circ(p, r, d):
    x, y = p
    bbox = [(x - r, y - r), (x + r, y + r)]
    d.ellipse(bbox, fill=(2, 0, 0))

def make_data():
    
    base = Image.new('RGBA', (Nx, Ny), (0, 0, 0))
    d = ImageDraw.Draw(base)

    rectx1, recty1, rectx2, recty2 = 0.15, 0.15, 0.85, 0.85
    rectx1 = rectx1 * Nx
    recty1 = (1 - recty1) * Ny
    rectx2 = rectx2 * Nx
    recty2 = (1 - recty2) * Ny

    rectbbox = [(rectx1, recty1), (rectx2, recty2)]
    d.rectangle(rectbbox, fill=(1, 0, 0))

    rs, ps = rand_circs()
    [make_circ(p, r, d) for p, r in zip([x for x in ps.T], rs)]
    
    base_array = np.array(base.getdata())
    material_array = base_array.reshape(Nx, Ny, 4)[:, :, 0]
    
    airidx = np.where(material_array == 0)
    m1idx = np.where(material_array == 1)
    m2idx = np.where(material_array == 2)
    
    m1_atts = 0.226
    m2_atts = 0.595
    atts = np.vstack([m1_atts, m2_atts]).T
    
    array = np.zeros_like(material_array, dtype=float)
    array[m1idx] = atts[0, 0]
    array[m2idx] = atts[0, 1]
    
    lim_sino = ray_trafo_lim(array)
    lim_recon = fbp_lim(lim_sino)
    lim_recon = lim_recon.asarray()
    
    full_sino = ray_trafo_full(array)
    full_recon = fbp_full(full_sino)
    full_recon = full_recon.asarray()
    
    return full_recon,lim_recon
