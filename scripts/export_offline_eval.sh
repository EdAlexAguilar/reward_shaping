cp_checkpoint=/home/luigi/Desktop/aaai22/checkpoints/cart_pole_obst/

outpath=/home/luigi/Desktop/aaai22/checkpoints/exports
n_eval_episodes=50

for reward in sparse stl weighted gb_chain gb_bpr_ci gb_bpdr_ci
do
  python reward_shaping/logging/eval_agent.py \
          --path $cp_checkpoint \
          --outpath $outpath \
          --regex "**/model*2000000*.zip" \
          --env cart_pole_obst \
          --train_reward $reward \
          --eval_reward offline_eval \
          --eval_episodes $n_eval_episodes \
          -no_render -save
done