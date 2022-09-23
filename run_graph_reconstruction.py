import os
import subprocess

tasks = ["md"]
dimension = [10]
# # dataset = ['california','grpc','road-m','web-edu']
# # dataset = ['cs_phd','web-edu','power','facebook','grpc', 'california','road-m', 'bio-diseasome','bio-wormnet']
dataset = ['grqc']
space_dims = [9,8,7,6,5,4,3,2,1,0]

def run_QGCN():
    for task in tasks:
        for dim in dimension:
            for data in dataset:
                for space_dim in space_dims:
                    lr = 0.01
                    command = "python train.py --task %s --dataset %s --model HGCN --lr %s --dim %d --num-layers 2 --act relu --bias 1 --dropout 0 --weight-decay 0 --manifold PseudoHyperboloid --log-freq 5 --cuda 0 --c None --space_dim %s --time_dim %s --epoch 2000" % (task, data, lr, dim, space_dim, dim-space_dim)
                    print(command)
                    process = subprocess.Popen(command, stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell = True)
                    stdout, stderr = process.communicate()
                    process.wait()
                    prefix = "./logs/md510/"+data+"/pseudo_hyperboloid_skip_lr" + "_" +  task + "_" + data + "_" + str(dim)+ "_" + str(space_dim)
                    stdout_name = prefix + ".out" 
                    stderr_nameå = prefix + ".err"
                    print(stdout, stderr)
                    with open(stdout_name, "w") as out, open(stderr_name, "w") as err:
                        out.write(stdout.decode("utf-8"))
                        err.write(stderr.decode("utf-8"))


# dataset = ['airport', 'pubmed', 'citeseer','cora']
# space_dims = [15,14,13,8,3,2,1]
# tasks = ['nc']
# models = ['HNN']

def run_QGCN_nc():
    for data in dataset:
        for task in tasks:
            for space_dim in space_dims:
                for model in models:
                    command = "python train.py --task %s --dataset %s --model %s --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.5 --weight-decay 0.0001 --manifold PseudoHyperboloid --log-freq 5 --cuda 0 --c None --space_dim %s --time_dim %s --epoch 1000" % (task, data, model,space_dim, 16-space_dim)
                    print(command)
                    process = subprocess.Popen(command, stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell = True)
                    stdout, stderr = process.communicate()
                    process.wait()
                    prefix = "./logs/nc/pseudo_hyperboloid_skip_HNN"+ model + "_" +  task + "_" + data + "_" + str(16)+ "_" + str(space_dim)
                    stdout_name = prefix + ".out" 
                    stderr_name = prefix + ".err"
                    print(stdout, stderr)
                    with open(stdout_name, "w") as out, open(stderr_name, "w") as err:
                        out.write(stdout.decode("utf-8"))
                        err.write(stderr.decode("utf-8"))

# tasks = ["md"]
# dimension = [10]
# # # dataset = ['california','grpc','road-m','web-edu']
# dataset = ['cs_phd','web-edu','power','facebook','grqc', 'california','road-m', 'bio-diseasome','bio-wormnet']

def run_Euclidean():
    for task in tasks:
        for dim in dimension:
            for data in dataset:
                command = "python train.py --task %s --dataset %s --model GCN --lr 0.01 --dim %d --num-layers 2 --act relu --bias 1 --dropout 0 --weight-decay 0 --manifold Euclidean --log-freq 1 --cuda 0 --epoch 1000" % (task, data, dim)
                print(command)
                process = subprocess.Popen(command, stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell = True)
                stdout, stderr = process.communicate()
                process.wait()

                prefix = "./logs/GCN"+ "_" + task + "_" + data + "_" + str(dim)
                stdout_name = prefix + ".out" 
                stderr_name = prefix + ".err"
                with open(stdout_name, "w") as out, open(stderr_name, "w") as err:
                    print(stderr)
                    out.write(stdout.decode("utf-8"))
                    err.write(stderr.decode("utf-8"))


def run_MLP():
    for task in tasks:
        for dim in dimension:
            for data in dataset:
                command = "python train.py --task %s --dataset %s --model MLP --lr 0.01 --dim %d --num-layers 2 --act relu --bias 1 --dropout 0 --weight-decay 0 --manifold Euclidean --log-freq 5 --cuda 0 " % (task, data, dim)
                # command = "python tran"
                print(command)
                process = subprocess.Popen(command, stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell = True)
                stdout, stderr = process.communicate()
                process.wait()

                prefix = "./logs/MLP"+ "_" + task + "_" + data + "_" + str(dim)
                stdout_name = prefix + ".out" 
                stderr_name = prefix + ".err"
                with open(stdout_name, "w") as out, open(stderr_name, "w") as err:
                    print(stderr)
                    out.write(stdout.decode("utf-8"))
                    err.write(stderr.decode("utf-8"))

def run_HNN():
    for task in tasks:
        for dim in dimension:
            for data in dataset:
                command = "python train.py --task %s --dataset %s --model HNN --lr 0.01 --dim %d --num-layers 2 --act relu --bias 1 --dropout 0 --weight-decay 0 --manifold Hyperboloid --log-freq 5 --cuda 1 --c 1.0" % (task, data, dim)
                # command = "python tran"
                print(command)
                process = subprocess.Popen(command, stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell = True)
                stdout, stderr = process.communicate()
                process.wait()

                prefix = "./logs/HNN"+ "_" + task + "_" + data + "_" + str(dim)
                stdout_name = prefix + ".out" 
                stderr_name = prefix + ".err"
                with open(stdout_name, "w") as out, open(stderr_name, "w") as err:
                    print(stderr)
                    out.write(stdout.decode("utf-8"))
                    err.write(stderr.decode("utf-8"))

def run_HGCN():
    for task in tasks:
        for dim in dimension:
            for data in dataset:
                for manifold in ["Hyperboloid", "PoincareBall"]:
                    command = "python train.py --task %s --dataset %s --model HGCN --lr 0.01 --dim %d --num-layers 2 --act relu --bias 1 --dropout 0 --weight-decay 0 --manifold %s --log-freq 5 --cuda 3" % (task, data, dim, manifold)
                    print(command)
                    process = subprocess.Popen(command, stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell = True)
                    stdout, stderr = process.communicate()
                    process.wait()

                    prefix = "./logs/" + manifold+ "_" + task + "_" + data + "_" + str(dim)
                    stdout_name = prefix + ".out" 
                    stderr_name = prefix + ".err"
                    print(stdout_name)
                    with open(stdout_name, "w") as out, open(stderr_name, "w") as err:
                        out.write(stdout.decode("utf-8"))
                        err.write(stderr.decode("utf-8"))

# dimension = [16]
# space_dims = [15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]

def run_HGCN_cora():
    for dim in dimension:
        for space_dim in space_dims:
            command = "python train.py --task nc --dataset citeseer --model HGCN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.5 --weight-decay 0.0001 --manifold PseudoHyperboloid --log-freq 5 --cuda 0 --c None --space_dim %s --time_dim %s" % (space_dim, dim-space_dim)
            print(command)
            process = subprocess.Popen(command, stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell = True)
            stdout, stderr = process.communicate()
            process.wait()

            prefix = "./nc/logs/" + 'pseudo'+ "_" + 'nc' + "_" + 'citeseer' + "_" + str(space_dim)
            stdout_name = prefix + ".out" 
            stderr_name = prefix + ".err"
            print(stdout_name)
            with open(stdout_name, "w") as out, open(stderr_name, "w") as err:
                out.write(stdout.decode("utf-8"))
                err.write(stderr.decode("utf-8"))

if __name__ == "__main__":
    
    #run_dummy_test()
    # run_Euclidean()
    # run_MLP()
    # run_HNN()
    run_QGCN()
    # run_QGCN_nc()
    # run_HGCN_cora()
    # run_disease()
    # run_Euclidean()
    # python train.py --task md --dataset road-m --model GCN --lr 0.01 --dim 10 --num-layers 2 --act relu --bias 1 --dropout 0 --weight-decay 0 --manifold Euclidean --log-freq 5 --cuda -1 
    # python train.py --task md --dataset bio-diseasome --model MLP --lr 0.01 --dim 10 --num-layers 2 --act relu --bias 1 --dropout 0 --weight-decay 0 --manifold Euclidean --log-freq 5 --cuda 0  97
    # python train.py --task md --dataset cs_phd --model MLP --lr 0.01 --dim 10 --num-layers 2 --act relu --bias 1 --dropout 0 --weight-decay 0 --manifold Euclidean --log-freq 5 --cuda 0
    # python train.py --task md --dataset road-m --model HGCN --lr 0.01 --dim 10 --num-layers 2 --act relu --bias 0 --dropout 0 --weight-decay 0 --manifold PseudoHyperboloid --log-freq 5 --cuda -1 --space_dim 9 --time_dim 1
    # python train.py --task md --dataset cycle_tree --model GCN --lr 0.01 --dim 10 --num-layers 2 --act relu --bias 1 --dropout 0 --weight-decay 0 --manifold Euclidean --log-freq 5 --cuda 1
    # python train.py --task md --dataset web-edu --model HGCN --lr 0.01 --dim 10 --num-layers 2 --act relu --bias 1 --dropout 0 --weight-decay 0 --manifold PseudoHyperboloid --log-freq 5 --space_dim 9 --time_dim 1 --cuda 1
    # python train.py --task md --dataset bio-wormnet --model HNN --lr 0.01 --dim 10 --num-layers 2 --act relu --bias 1 --dropout 0 --weight-decay 0 --manifold PseudoHyperboloid --log-freq 5 --space_dim 8 --time_dim 2 --cuda 1


# python train.py --task md --dataset california --model GCN --lr 0.01 --dim 10 --num-layers 2 --act relu --bias 1 --dropout 0 --weight-decay 0 --manifold Euclidean --log-freq 5 --cuda 0

# python train.py --task md --dataset cs_phd --model HNN --lr 0.01 --dim 10 --num-layers 2 --act relu --bias 1 --dropout 0 --weight-decay 0 --manifold Hyperboloid --log-freq 5 --cuda 1 --c 1.0
# python train.py --task md --dataset tree_cycle --model MLP --lr 0.01 --dim 10 --num-layers 2 --act relu --bias 1 --dropout 0 --weight-decay 0 --manifold Euclidean --log-freq 5 --cuda -1 
# python train.py --task nc --dataset cora --model GCN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0 --weight-decay 0 --manifold Euclidean --log-freq 5 --cuda 0 

# python train.py --task nc --dataset cora --model HNN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.5 --weight-decay 0.0001 --manifold PseudoHyperboloid --log-freq 5 --cuda 0 --c None --space_dim 15 --time_dim 1

# python train.py --task lp --dataset disease_lp --model HGCN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.5 --weight-decay 0.0001 --manifold PseudoHyperboloid --log-freq 5 --cuda 0 --c None --space_dim 15 --time_dim 1
# python train.py --task md --dataset cs_phd --model HGCN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.5 --weight-decay 0.0001 --manifold PseudoHyperboloid --log-freq 5 --cuda 0 --c None --space_dim 15 --time_dim 1

# python train.py --task nc --dataset cora --model  GCN --lr 0.01 --dim 10 --num-layers 2 --act relu --bias 1 --dropout 0 --weight-decay 0 --manifold Euclidean --log-freq 5 --cuda 2

# python train.py --task md --dataset cycle_tree --model HGCN --lr 0.01 --dim 10 --num-layers 2 --act relu --bias 1 --dropout 0 --weight-decay 0 --manifold PseudoHyperboloid --log-freq 5 --cuda 0 --c None --space_dim 7 --time_dim 3
           
# python train.py --task lp --dataset cora --model GCN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.2 --weight-decay 0 --manifold Euclidean --log-freq 5 --cuda 0

# python train.py --task nc --dataset cora --model HGCN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.5 --weight-decay 0.0001 --manifold PseudoHyperboloid --log-freq 5 --cuda 0 --c None --space_dim 15 --time_dim 1

# python train.py --task nc --dataset cora --model HGCN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.5 --weight-decay 0.0001 --manifold PseudoHyperboloid --log-freq 5 --cuda 0 --c None --space_dim 15 --time_dim 1


# python train.py --task md --dataset cs_phd --model GCN --lr 0.01 --dim 10 --num-layers 2 --act relu --bias 1 --dropout 0 --weight-decay 0 --manifold Euclidean --log-freq 5 --cuda 0
# python train.py --task md --dataset bio-diseasome --model GCN --lr 0.01 --dim 10 --num-layers 2 --act relu --bias 1 --dropout 0 --weight-decay 0 --manifold Euclidean --log-freq 5 --cuda 0


# python train.py --task md --dataset bio-diseasome --model GCN --lr 0.01 --dim 10 --num-layers 2 --act relu --bias 1 --dropout 0 --weight-decay 0 --manifold Euclidean --log-freq 5 --cuda 0

# python train.py --task md --dataset bio-diseasome --model HGCN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.5 --weight-decay 0.0001 --manifold PseudoHyperboloid --log-freq 5 --cuda 0 --c None --space_dim 15 --time_dim 1

# python train.py --task md --dataset cs_phd --model HGCN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.5 --weight-decay 0.0001 --manifold PseudoHyperboloid --log-freq 5 --cuda 0 --c None --space_dim 15 --time_dim 1