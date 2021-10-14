# Quantum-Computation-Project
The code files of the project I performed in Quantum Computation, under the guidance of Shira Chapman.

Here you can find an explaination of how to run the project files properly.

The 'whole_process.py' module contains the whole process (surprisingly) of generating a TFD state given a value for beta (1/T).
It imports the functions 'circ25' and 'circ25_noMeasurements_forFidelity' from the module 'util.py', which in turn recieves a TFD state, a temperature (in the form of beta) and an initial state for q0 (in the form of a string, see module), runs the circuit of the experiment on the wanted simulator and returns the results.

In 'whole_process.py', the function 'run_exp' and 'run_exp_fid' perform the whole process of altering the temperature in a wanted range (via beta), performing the experiment and writing the results in a .csv file.

From these results, we later generate plots and fits using the 'plots.py' and 'fits.py' modules, in which the relevant functions compute the relevant figures, save them and show them (notice that not all save, and not all show for comfort reasons - for example running the experiment for 1000 beta's --> Don't want to stop the computation each iteration).

That's quite it. I have not uploaded to this page all of he data I have collected (the data used for the project graphs) since it's quite a lot. If interested in it, may freely email me at razmon@post.bgu.ac.il or at razmonsonego2@gmail.com, and I'll hapilly transfer the files by request.

Have fun :)
