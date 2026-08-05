# submit.sh
ssh usuario@hpc '
  cd /home/usuario/heavy-ion-sim &&
  condor_submit submit/condor.sub
'