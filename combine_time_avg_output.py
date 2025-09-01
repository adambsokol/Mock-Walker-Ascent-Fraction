"""
April 22, 2025

Postprocessing scipt that calls a function from postprocess.py.
Combined the time-averaged MSE output files from the different cases within a set into a single file.
"""
import glob
import postprocess
import numpy as np

SSTs = [298, 300, 302, 304, 306, 308]

# Fixed 4
print('Fixed 4')
t0 = '150'
t1 = '300'
flist = [f'OUT_MSE/MEAN/fixed_{sst}d4_nx3072_d{t0}-{t1}.nc' for sst in SSTs]
flist.sort()
flist[0] = 'OUT_MSE/MEAN/fixed_298d4_nx3072_d350-500.nc'
new_name = f'fixed4_nx3072_d{t0}-{t1}.nc'
postprocess.combine_time_averaged_output(flist, new_name)

# Fixed 3
print('Fixed 3')
t0 = '150'
t1 = '300'
flist = [f'OUT_MSE/MEAN/fixed_{sst}d3_nx3072_d{t0}-{t1}.nc' for sst in SSTs]
flist.sort()
flist[0] = 'OUT_MSE/MEAN/fixed_298d3_nx3072_d300-450.nc'
new_name = f'fixed3_nx3072_d{t0}-{t1}.nc'
postprocess.combine_time_averaged_output(flist, new_name)
"""
# Fixed 2
print('Fixed 2')
t0 = '150'
t1 = '300'
flist = [f'OUT_MSE/MEAN/fixed_{sst}d2_nx3072_d{t0}-{t1}.nc' for sst in SSTs]
flist.sort()
new_name = f'fixed2_nx3072_d{t0}-{t1}.nc'
postprocess.combine_time_averaged_output(flist, new_name)

# Fixed 8
print('Fixed 8')
t0 = '150'
t1 = '300'
flist = [f'OUT_MSE/MEAN/fixed_{sst}d8_nx3072_d{t0}-{t1}.nc' for sst in SSTs]
flist.sort()
new_name = f'fixed8_nx3072_d{t0}-{t1}.nc'
postprocess.combine_time_averaged_output(flist, new_name)
"""
# slab100
print('Slab 100')
t0 = '150'
t1 = '300'
flist = [
    f'OUT_MSE/MEAN/slabmw_{sst}d4_ds100_nx3072_d{t0}-{t1}.nc' for sst in SSTs]
flist.sort()
new_name = f'slabmw_ds100_nx3072_d{t0}-{t1}.nc'
postprocess.combine_time_averaged_output(flist, new_name)

# slab70
print('Slab 70')
t0 = '150'
t1 = '300'
flist = [
    f'OUT_MSE/MEAN/slabmw_{sst}d4_ds70_nx3072_d{t0}-{t1}.nc' for sst in SSTs]
flist.sort()
new_name = f'slabmw_ds70_nx3072_d{t0}-{t1}.nc'
postprocess.combine_time_averaged_output(flist, new_name)
"""
# slab60
print('Slab 60')
t0 = '150'
t1 = '300'
flist = [
    f'OUT_MSE/MEAN/slabmw_{sst}d4_ds60_nx3072_d{t0}-{t1}.nc' for sst in SSTs]
flist.sort()
new_name = f'slabmw_ds60_nx3072_d{t0}-{t1}.nc'
postprocess.combine_time_averaged_output(flist, new_name)

# slab50
print('Slab 50')
t0 = '150'
t1 = '300'
flist = [
    f'OUT_MSE/MEAN/slabmw_{sst}d4_ds50_nx3072_d{t0}-{t1}.nc' for sst in SSTs]
flist.sort()
new_name = f'slabmw_ds50_nx3072_d{t0}-{t1}.nc'
postprocess.combine_time_averaged_output(flist, new_name)

# slab40
print('slab 40')
t0 = '150'
t1 = '300'
flist = [
    f'OUT_MSE/MEAN/slabmw_{sst}d4_ds40_nx3072_d{t0}-{t1}.nc' for sst in SSTs]
flist.sort()
new_name = f'slabmw_ds40_nx3072_d{t0}-{t1}.nc'
postprocess.combine_time_averaged_output(flist, new_name)

# slab35
print('slab 35')
t0 = '150'
t1 = '300'
flist = [
    f'OUT_MSE/MEAN/slabmw_{sst}d4_ds35_nx3072_d{t0}-{t1}.nc' for sst in SSTs]
flist.sort()
new_name = f'slabmw_ds35_nx3072_d{t0}-{t1}.nc'
postprocess.combine_time_averaged_output(flist, new_name)
"""
# slab20
print('slab 20')
t0 = '150'
t1 = '300'
flist = [
    f'OUT_MSE/MEAN/slabmw_{sst}d4_ds20_nx3072_d{t0}-{t1}.nc' for sst in SSTs]
flist.sort()
new_name = f'slabmw_ds20_nx3072_d{t0}-{t1}.nc'
postprocess.combine_time_averaged_output(flist, new_name)
"""
# slab0
print('slab 0')
t0 = '150'
t1 = '300'
flist = [
    f'OUT_MSE/MEAN/slabf_{sst}d4_nx3072_d{t0}-{t1}.nc' for sst in SSTs]
flist.sort()
new_name = f'slab0_nx3072_d{t0}-{t1}.nc'
postprocess.combine_time_averaged_output(flist, new_name)
"""

########## 3D experiments ##########
"""
# fixed4 3D experiments
print('Fixed 4 3D')
t0 = '75'
t1 = '150'
flist = [f'OUT_MSE/MEAN/fixed3d_{sst}d4_nx3072_d{t0}-{t1}.nc' for sst in SSTs]
flist.sort()
new_name = f'fixed3d_nx3072_d{t0}-{t1}.nc'
postprocess.combine_time_averaged_output(flist, new_name)

# slab0 3D experiments
print('Slab0 3D')
t0 = '100'
t1 = '200'
flist = [f'OUT_MSE/MEAN/slabf3d_{sst}d4_nx3072_d{t0}-{t1}.nc' for sst in SSTs]
flist.sort()
new_name = 'slabf3d_nx3072_d100-200.nc'
postprocess.combine_time_averaged_output(flist, new_name)

# slab70 3D experiments
print('Slab70 3D')
t0 = '100'
t1 = '250'
flist = [
    f'OUT_MSE/MEAN/slabmw3d_{sst}d4_ds70_nx3072_d{t0}-{t1}.nc' for sst in SSTs]
flist.sort()
new_name = f'slabmw3d_ds70_nx3072_d{t0}-{t1}.nc'
postprocess.combine_time_averaged_output(flist, new_name)


########## NX=1536 experiments ##########
# Fixed 4
print('Fixed 4 nx1536')
t0 = '150'
t1 = '300'
flist = [f'OUT_MSE/MEAN/fixed_{sst}d4_nx1536_d{t0}-{t1}.nc' for sst in SSTs]
flist.sort()
new_name = f'fixed4_nx1536_d{t0}-{t1}.nc'
postprocess.combine_time_averaged_output(flist, new_name)

# Fixed 4
print('Fixed 2 nx1536')
t0 = '150'
t1 = '300'
flist = [f'OUT_MSE/MEAN/fixed_{sst}d2_nx1536_d{t0}-{t1}.nc' for sst in SSTs]
flist.sort()
new_name = f'fixed2_nx1536_d{t0}-{t1}.nc'
postprocess.combine_time_averaged_output(flist, new_name)

# Fixed 4
print('Slab 35 nx1536')
t0 = '150'
t1 = '300'
flist = [
    f'OUT_MSE/MEAN/slabmw_{sst}d2_ds35_nx1536_d{t0}-{t1}.nc' for sst in SSTs]
flist.sort()
new_name = f'slabmw_ds35_nx1536_d{t0}-{t1}.nc'
postprocess.combine_time_averaged_output(flist, new_name)

# slab70
print('Slab 70 nx1536')
t0 = '150'
t1 = '300'
flist = [
    f'OUT_MSE/MEAN/slabmw_{sst}d4_ds70_nx1536_d{t0}-{t1}.nc' for sst in SSTs]
flist.sort()
new_name = f'slabmw_ds70_nx1536_d{t0}-{t1}.nc'
postprocess.combine_time_averaged_output(flist, new_name)


########## NX=4320 experiments ##########
# Fixed 4
print('Fixed 4 nx4320')
t0 = '150'
t1 = '300'
flist = [f'OUT_MSE/MEAN/fixed_{sst}d4_d{t0}-{t1}.nc' for sst in SSTs]
flist.sort()
new_name = f'fixed4_nx4320_d{t0}-{t1}.nc'
postprocess.combine_time_averaged_output(flist, new_name)


# slab70
print('Slab 70 nx4320')
t0 = '150'
t1 = '300'
flist = [
    f'OUT_MSE/MEAN/slabmw_{sst}d4_ds70_d{t0}-{t1}.nc' for sst in SSTs]
flist.sort()
new_name = f'slabmw_ds70_nx4320_d{t0}-{t1}.nc'
postprocess.combine_time_averaged_output(flist, new_name)



########## NX=4608 experiments ##########
# Fixed 6
print('Fixed 6 nx4608')
t0 = '150'
t1 = '300'
flist = [f'OUT_MSE/MEAN/fixed_{sst}d6_nx4608_d{t0}-{t1}.nc' for sst in SSTs]
flist.sort()
new_name = f'fixed6_nx4608_d{t0}-{t1}.nc'
postprocess.combine_time_averaged_output(flist, new_name)

# slab105 w/ dT=6
print('Slab 105 nx4608')
t0 = '150'
t1 = '300'
flist = [
    f'OUT_MSE/MEAN/slabmw_{sst}d6_ds105_nx4608_d{t0}-{t1}.nc' for sst in SSTs]
flist.sort()
new_name = f'slabmw_ds105_nx4608_d{t0}-{t1}.nc'
postprocess.combine_time_averaged_output(flist, new_name)
"""
# slab105 w/ dT=6
print('Slab 70 nx4608')
t0 = '150'
t1 = '300'
flist = [
    f'OUT_MSE/MEAN/slabmw_{sst}d4_ds70_nx4608_d{t0}-{t1}.nc' for sst in SSTs]
flist.sort()
new_name = f'slabmw_ds70_nx4608_d{t0}-{t1}.nc'
postprocess.combine_time_averaged_output(flist, new_name)
"""
########## NX=6144 experiments ##########
# Fixed 8
print('Fixed 8 nx6144')
t0 = '150'
t1 = '300'
flist = [f'OUT_MSE/MEAN/fixed_{sst}d8_nx6144_d{t0}-{t1}.nc' for sst in SSTs]
flist.sort()
new_name = f'fixed8_nx6144_d{t0}-{t1}.nc'
postprocess.combine_time_averaged_output(flist, new_name)
"""
