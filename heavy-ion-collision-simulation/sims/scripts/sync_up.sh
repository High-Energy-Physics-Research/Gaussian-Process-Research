# sync_up.sh
rsync -av --delete \
  --exclude ".git" \
  --exclude "output/" \
  --exclude "logs/" \
  ./ usuario@hpc:/home/usuario/heavy-ion-sim/