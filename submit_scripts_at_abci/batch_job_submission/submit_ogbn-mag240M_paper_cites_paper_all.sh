GROUP_ID=gac50544
qsub -g $GROUP_ID -o ogbn-mag240M_paper_cites_paper/log/ -e ogbn-mag240M_paper_cites_paper/log/ ./run_ogbn-mag240M_paper_cites_paper_16.sh
qsub -g $GROUP_ID -o ogbn-mag240M_paper_cites_paper/log/ -e ogbn-mag240M_paper_cites_paper/log/ ./run_ogbn-mag240M_paper_cites_paper_32.sh
qsub -g $GROUP_ID -o ogbn-mag240M_paper_cites_paper/log/ -e ogbn-mag240M_paper_cites_paper/log/ ./run_ogbn-mag240M_paper_cites_paper_64.sh
qsub -g $GROUP_ID -o ogbn-mag240M_paper_cites_paper/log/ -e ogbn-mag240M_paper_cites_paper/log/ ./run_ogbn-mag240M_paper_cites_paper_128.sh
qsub -g $GROUP_ID -o ogbn-mag240M_paper_cites_paper/log/ -e ogbn-mag240M_paper_cites_paper/log/ ./run_ogbn-mag240M_paper_cites_paper_256.sh
