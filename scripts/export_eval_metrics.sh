cp_logdir=/home/luigi/Desktop/aaai22/logs/cart_pole_obst/
bw_logdir=/home/luigi/Desktop/aaai22/logs/bipedal_walker/
ll_logdir=/home/luigi/Desktop/aaai22/logs/lunar_lander/

metric='eval'

for logdir in $cp_logdir $bw_logdir $ll_logdir
do
  python reward_shaping/logging/export_data.py \
          --path ${logdir} \
          --outpath ${logdir}/${metric}_exports \
          --tag ${metric}
done