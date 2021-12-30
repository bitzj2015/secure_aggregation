python main.py --subset 1 --version lin_1_avg --model linear &> ./param/log_lin_1_avg.txt
python main.py --subset 2 --version lin_2_avg --model linear &> ./param/log_lin_2_avg.txt
python main.py --subset 1 --version nn_1_avg --model fcnn &> ./param/log_nn_1_avg.txt
python main.py --subset 2 --version nn_2_avg --model fcnn &> ./param/log_nn_2_avg.txt

# python main.py --subset 5 --version nn_5 --model fcnn &> ./param/log_nn_5.txt
# python main.py --subset 10 --version nn_10 --model fcnn &> ./param/log_nn_10.txt
# python main.py --subset 20 --version nn_20 --model fcnn &> ./param/log_nn_20.txt