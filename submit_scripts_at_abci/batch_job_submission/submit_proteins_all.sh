GROUP_ID=gac50544
qsub -g $GROUP_ID -o proteins/log/ -e proteins/log/ ./run_proteins_2.sh
qsub -g $GROUP_ID -o proteins/log/ -e proteins/log/ ./run_proteins_4.sh
qsub -g $GROUP_ID -o proteins/log/ -e proteins/log/ ./run_proteins_8.sh
qsub -g $GROUP_ID -o proteins/log/ -e proteins/log/ ./run_proteins_16.sh
qsub -g $GROUP_ID -o proteins/log/ -e proteins/log/ ./run_proteins_32.sh
qsub -g $GROUP_ID -o proteins/log/ -e proteins/log/ ./run_proteins_64.sh
qsub -g $GROUP_ID -o proteins/log/ -e proteins/log/ ./run_proteins_128.sh
qsub -g $GROUP_ID -o proteins/log/ -e proteins/log/ ./run_proteins_256.sh
