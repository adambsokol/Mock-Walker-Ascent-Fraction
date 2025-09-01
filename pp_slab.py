"""
April 22, 2025

Postprocessing scipt that calls a function from postprocess.py.
Combined the time-averaged MSE output files from the different cases within a set into a single file.
"""
import glob
import postprocess

#########################################
#########################################
# Slab with varying ocean heat source S

print('slab with varying S')
"""
case = 'slabmw_300d4_s35_ds70_nx3072'
# postprocess.combine_MSE_output(case)
# postprocess.create_time_averaged_output(
#    case, avg_period=0, tstart=1250, tend=1500)

case = 'slabmw_300d4_s33_ds70_nx3072'
# postprocess.combine_MSE_output(case)
# postprocess.create_time_averaged_output(
#    case, avg_period=0, tstart=1750, tend=2000)
postprocess.create_time_averaged_output(
    case, avg_period=0, tstart=1050, tend=1300)

case = 'slabmw_302d4_s285_ds70_nx3072'
# postprocess.combine_MSE_output(case)
#postprocess.create_time_averaged_output(
#    case, avg_period=0, tstart=1750, tend=2000)



case = 'slabmw_300d4_s34_ds70_nx3072'
# postprocess.combine_MSE_output(case)
postprocess.create_time_averaged_output(
    case, avg_period=0, tstart=1250, tend=1500)
postprocess.create_time_averaged_output(
    case, avg_period=0, tstart=650, tend=900)

case = 'slabmw_302d4_s315_ds70_nx3072'
# postprocess.combine_MSE_output(case)
postprocess.create_time_averaged_output(
    case, avg_period=0, tstart=250, tend=500)

case = 'slabmw_300d4_s30_ds70_nx3072'
# postprocess.combine_MSE_output(case)
# postprocess.create_time_averaged_output(
#    case, avg_period=0, tstart=2250, tend=2500)
# postprocess.create_time_averaged_output(
#    case, avg_period=0, tstart=2000, tend=2250)
# postprocess.create_time_averaged_output(
#    case, avg_period=0, tstart=600, tend=850)
postprocess.create_time_averaged_output(
    case, avg_period=0, tstart=1950, tend=2200)


case = 'slabmw_302d4_s27_ds70_nx3072'
# postprocess.combine_MSE_output(case)
# postprocess.create_time_averaged_output(
#    case, avg_period=0, tstart=1250, tend=1500)
postprocess.create_time_averaged_output(
    case, avg_period=0, tstart=800, tend=1050)
postprocess.create_time_averaged_output(
    case, avg_period=0, tstart=1750, tend=2000)

flist = [
    f'OUT_MSE/MEAN/slabmw_300d4_s34_ds70_nx3072_d1250-1500.nc',
    f'OUT_MSE/MEAN/slabmw_300d4_s34_ds70_nx3072_d650-900.nc',
    f'OUT_MSE/MEAN/slabmw_302d4_s315_ds70_nx3072_d250-500.nc',
    f'OUT_MSE/MEAN/slabmw_300d4_s30_ds70_nx3072_d1950-2200.nc',
    f'OUT_MSE/MEAN/slabmw_302d4_s27_ds70_nx3072_d800-1050.nc',
    f'OUT_MSE/MEAN/slabmw_302d4_s27_ds70_nx3072_d1750-2000.nc',
]
new_name = 'slabS_ds70_nx3072_last250_every2K.nc'
postprocess.combine_time_averaged_output(flist, new_name)
"""

#########################################
#########################################
# Slab with varying CO2
print('slab with varying CO2')

case = 'slabmw_300d4_s315_ds70_0p7xCO2_nx3072'
#postprocess.combine_MSE_output(case)
postprocess.create_time_averaged_output(
    case, avg_period=0, tstart=250, tend=500)


case = 'slabmw_302d4_s315_ds70_0p9xCO2_nx3072'
#postprocess.combine_MSE_output(case)
postprocess.create_time_averaged_output(
    case, avg_period=0, tstart=850, tend=1100)

case = 'slabmw_302d4_s315_ds70_nx3072'
#postprocess.combine_MSE_output(case)
postprocess.create_time_averaged_output(
    case, avg_period=0, tstart=1250, tend=1500)

case = 'slabmw_304d4_s315_ds70_1p1xCO2_nx3072'
#postprocess.combine_MSE_output(case)
postprocess.create_time_averaged_output(
    case, avg_period=0, tstart=1250, tend=1500)


case = 'slabmw_304d4_s315_ds70_1p4xCO2_nx3072'
#postprocess.combine_MSE_output(case)
postprocess.create_time_averaged_output(
    case, avg_period=0, tstart=1750, tend=2000)


case = 'slabmw_308d4_s315_ds70_1p8xCO2_nx3072'
#postprocess.combine_MSE_output(case)
postprocess.create_time_averaged_output(
    case, avg_period=0, tstart=1250, tend=1500)


flist = [
    f'OUT_MSE/MEAN/slabmw_300d4_s315_ds70_0p7xCO2_nx3072_d250-500.nc',
    f'OUT_MSE/MEAN/slabmw_302d4_s315_ds70_0p9xCO2_nx3072_d850-1100.nc',
    f'OUT_MSE/MEAN/slabmw_302d4_s315_ds70_nx3072_d1250-1500.nc',
    f'OUT_MSE/MEAN/slabmw_304d4_s315_ds70_1p1xCO2_nx3072_d1250-1500.nc',
    f'OUT_MSE/MEAN/slabmw_304d4_s315_ds70_1p4xCO2_nx3072_d1750-2000.nc',
    f'OUT_MSE/MEAN/slabmw_308d4_s315_ds70_1p8xCO2_nx3072_d1250-1500.nc',
]

new_name = 'slabCO2_ds70_nx3072_last250_every2K.nc'
postprocess.combine_time_averaged_output(flist, new_name)



"""
case = 'slabmw_302d4_s315_ds70_0p8xCO2_nx3072'
# postprocess.combine_MSE_output(case)
postprocess.create_time_averaged_output(
    case, avg_period=0, tstart=750, tend=1000)

case = 'slabmw_306d4_s315_ds70_1p6xCO2_nx3072'
# postprocess.combine_MSE_output(case)
postprocess.create_time_averaged_output(
    case, avg_period=0, tstart=750, tend=1000)


case = 'slabmw_306d4_s315_ds70_2xCO2_nx3072'
# postprocess.combine_MSE_output(case)
postprocess.create_time_averaged_output(
    case, avg_period=0, tstart=750, tend=1000)
"""
