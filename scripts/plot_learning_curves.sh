

cp_exports=/home/luigi/Desktop/aaai22/cart_pole_obst/exports/
bw_exports=/home/luigi/Desktop/aaai22/bipedal_walker/exports/
ll_exports=/home/luigi/Desktop/aaai22/lunar_lander/exports/

outpath=/home/luigi/Desktop/aaai22/
binning=40000

python reward_shaping/logging/plot_data.py \
        --path $cp_exports \
        --outpath $outpath \
        --binning $binning \
        --ylabel "Evaluation Metric" \
        --title "Cart Pole" \
        --clipy 0.0 1.75 \
        --ylim 0.0 1.8 \
        --hlines 1.5 \
        --rewards default sparse stl weighted gb_chain gb_bpdr_ci \
        -save

python reward_shaping/logging/plot_data.py \
        --path $ll_exports \
        --outpath $outpath \
        --binning $binning \
        --ylabel "Evaluation Metric" \
        --title "Lunar Lander" \
        --clipy 0.0 1.75 \
        --ylim 0.0 1.8 \
        --hlines 1.5 \
        --rewards default sparse stl weighted gb_chain gb_bpdr_ci \
        -save

python reward_shaping/logging/plot_data.py \
        --path $bw_exports \
        --outpath $outpath \
        --binning $binning \
        --ylabel "Evaluation Metric" \
        --title "Bipedal Walker" \
        --clipy 0.0 1.75 \
        --ylim 0.0 1.8 \
        --hlines 1.5 \
        --rewards default sparse stl weighted gb_chain gb_bpr_ci \
        -save

