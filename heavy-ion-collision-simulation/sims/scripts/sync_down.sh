# sync_down.sh
rsync -av \
  usuario@hpc:/home/usuario/heavy-ion-sim/output/ ./output/

rsync -av \
  usuario@hpc:/home/usuario/heavy-ion-sim/logs/ ./logs/