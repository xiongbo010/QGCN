from .base import ManifoldParameter
from .euclidean import Euclidean
from .hyperboloid import Hyperboloid
from .poincare import PoincareBall
from .pseudohyperboloid_sr import PseudoHyperboloid
# from .sphere import Sphere

# DGLBACKEND=pytorch dglke_train --dataset hetionet --data_path ./hetionet --data_files hetionet_train.tsv hetionet_valid.tsv hetionet_test.tsv --format 'raw_udd_hrt' --model_name TransE_l2 --batch_size 512 --neg_sample_size 256 --hidden_dim 200 --gamma 12.0 --lr 0.1 --max_step 100000 --log_interval 1000 --batch_size_eval 16 -adv --regularization_coef 1.00E-07 --test --num_thread 1 --gpu 0 1 2 3 --num_proc 8 --neg_sample_size_eval 10000 --async_update
# DGLBACKEND=pytorch dglke_train --dataset pharmkg --data_path ./pharmkg --data_files pharmkg_train.tsv pharmkg_valid.tsv pharmkg_test.tsv --format 'raw_udd_hrt' --model_name TransE_l2 --batch_size 512 --neg_sample_size 256 --hidden_dim 200 --gamma 12.0 --lr 0.1 --max_step 100000 --log_interval 1000 --batch_size_eval 16 -adv --regularization_coef 1.00E-07 --test --num_thread 1 --gpu 0 1 2 3 --num_proc 8 --neg_sample_size_eval 10000 --async_update