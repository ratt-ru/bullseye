#!/bin/bash
MAX_MB=$1
ANTENNA_CONF=EVLA
MODEL_SKY=$2
OUTPUT_PATH=$6
NOISE_STDDEV=$3
RA=$4
DEC=$5
if [ "$#" -ne 6 ]; then
    echo "Expected arguments to simulation script: maximum_MiB_used_for_visibilities (will try and get as close as possible to this) model_sky noise_stddev RA DEC output_path"
    exit 1
fi
TURBO_SIM_PATH=${MEQTREES_CATTERY_PATH}/Siamese/turbo-sim.py
BATCH_CONFIG=${MEQTREES_CATTERY_PATH}/Siamese/
echo ${TURBO_SIM_PATH}

if [ ! -f ${TURBO_SIM_PATH} ]; then
    echo "I can't find the turbo-sim in the Cattery! Make sure Meqtrees is installed and the env variable 'MEQTREES_CATTERY_PATH' is set"
    exit 1
fi
VIS_SIZE=8 #32bit complex
BANDS=8
NCHAN=$(echo ${BANDS}*64 | bc)
CORR=4
ANT=27
BL=$(echo "${ANT}*(${ANT}-1)/2+${ANT}" | bc)

bytes_per_timestep=$(echo ${VIS_SIZE}*${NCHAN}*${CORR}*${BL} | bc)
timesteps_needed=$(echo ${MAX_MB}*1024*1024/${bytes_per_timestep} | bc)
bytes_used=$(echo ${timesteps_needed}*${bytes_per_timestep} | bc)
mb_used=$(echo "scale=3; ${bytes_used}/1024/1024" | bc)
ms_name=${OUTPUT_PATH}/SIM.${mb_used}MiB.MS
echo ${mb_used} MiB being simulated

#write out a parset file for makems
printf "NParts=1\n\
NBands=${BANDS}\n\
NFrequencies=${NCHAN}\n\
StartFreq=1170e6\n\
StepFreq=10\n\
StartTime=2000/08/03/13:22:30\n\
StepTime=120\n\
NTimes=${timesteps_needed}\n\
RightAscension=${RA}\n\
Declination=${DEC}\n\
WriteAutoCorr=T\n\
AntennaTableName=${ANTENNA_CONF}\n\
MSName=${ms_name}\n\
VDSPath=${OUTPUT_PATH}" > ${OUTPUT_PATH}/parset.conf
echo Parset file generated... Invoking makems
makems ${OUTPUT_PATH}/parset.conf
echo Okay empty measurement set created.. Creating simulation parset file

printf "[turbo-sim:job]\n\
me.e_enable=0\n\
me.g_enable=0\n\
me.l_enable=0\n\
me.ncorr_enable=0\n\
me.p_enable=0\n\
me.sky.siamese_agw_azel_sky=0\n\
me.sky.siamese_oms_fitsimage_sky=0\n\
me.sky.siamese_oms_gridded_sky=0\n\
me.sky.siamese_oms_transient_sky=0\n\
me.sky.tiggerskymodel=1\n\
me.use_jones_inspectors=1\n\
me.use_skyjones_visualizers=0\n\
me.use_smearing=0\n\
me.z_enable=0\n\
ms_sel.ddid_index=0\n\
ms_sel.field_index=0\n\
ms_sel.ms_corr_sel=2x2\n\
ms_sel.ms_ifr_subset_str=all\n\
ms_sel.ms_polarization=XX XY YX YY\n\
ms_sel.ms_taql_str=None\n\
ms_sel.msname=${ms_name}_p0\n\
ms_sel.output_column=DATA\n\
ms_sel.select_channels=0\n\
ms_sel.tile_size=4\n\
noise_from_sefd=0\n\
noise_stddev=${NOISE_STDDEV}\n\
random_seed=time\n\
read_ms_model=0\n\
run_purr=0\n\
sim_mode=sim only\n\
tensormeqmaker.psv_class=PSVTensor\n\
tiggerlsm.filename=${MODEL_SKY}\n\
tiggerlsm.lsm_subset=all\n\
tiggerlsm.null_subset=None\n\
tiggerlsm.solvable_sources=0\n\
uvw_refant=default\n\
uvw_source=from MS" > ${OUTPUT_PATH}/turbo-sim.conf
echo Okay... simulation setting saved, now fire up the simulation...
meqtree-pipeliner.py -c ${OUTPUT_PATH}/turbo-sim.conf \[turbo-sim:job\] ${TURBO_SIM_PATH} =simulate
echo Okay... your dataset is done lets make an image
python /scratch/ska-bullseye/bullseye/bullseye/bullseye.py ${ms_name}_p0 --output_prefix ${OUTPUT_PATH}/img --npix_l 1024 --npix_m 1024 --cell_l 5.2734 --cell_m 5.2734 --pol I --conv_sup 4 --conv_oversamp 63 --output_format fits --field_id 0 --data_column "DATA" --open_default_viewer 1 --average_all 1 --use_back_end CPU
