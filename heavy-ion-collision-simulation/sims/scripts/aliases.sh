alias condor='ssh usuario@cluster condor_q'

alias qh='ssh usuario@cluster squeue -u \$USER'


alias hpctop='ssh -t usuario@cluster htop'

alias syncup='rsync -av ./ usuario@cluster:/home/user/project/'

alias syncdown='rsync -av usuario@cluster:/home/user/project/output ./output'

alias submit='ssh usuario@cluster "cd /home/user/project && condor_submit submit/job.sub"'

alias logh='ssh usuario@cluster "tail -f /home/user/project/logs/job.out"'