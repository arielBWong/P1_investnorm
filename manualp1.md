## how to run p1_mainscript

experiment parameters are saved in p/zdt_problems_hvnd.json files

single_run() can be used to debug or demonstrate step by step processes

para_run() launches 29 seeds for each problem/setting over 14 processes

## commit out plots for runs 
method plot_process(ax, target_problem, train_y, norm_train_y, denormalize, True, krg, train_x)
needs to be set false( set visual to false) if only experimental results are needed.
There are multiple calls of it:

step(0) get ready to plot
 
step (4-0) before enter propose x phase, conduct once krg search on ideal

step (4-1) DE search for proposing next x point

step (4-2) according to configuration determine whether to estimate new point
