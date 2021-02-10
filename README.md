# Mean-field-optimal-control-for-biological-pattern-formation

This file explains how to use the parameter identification code corresponding to the submission: arXiv:2009.09894 

- init.py
  main parameters for the parameter identification are fixed
  the initial choice of parameters eta and theta are set
  Options 1-16 are provided for the initial choice of eta and theta, in a similar way other input values for eta and theta can be set
  The specific choice of eta and theta is selected via the input argument of main.py through 1-16
  folder path to results is set
  init.py is saved to output folder so that you can always remember the parameter setting

- main.py
  contains the main programm. 
  - FDAD allows you to check the gradient of the parameter identification algorithm
  - line_search is called to realise the line search inside the parameter identification algorithm
  - Jappend is called in the parameter identification algorithm and stores the current value of the cost functional for postprocessing
  - optimize is the main part of the parameter identification algorithm, it calls all other files and functions except for the postprocessing
  - compute_ref_solution computes one solution of the forward problem for given parameters. The results are used as artificial data for the parameter identification. x- and y-components of the solution are saved separately as 'x_opt.txt' and 'y_opt.txt' to the output folder

- forward.py
  computes the solution of the forward problem for given initial data and controls
  - interaction_force computes the pairwise interaction forces in a vectorized manner and takes the periodicity of the domain into account
  - if save flag is active, the forward solution is saved to path-folder with x- and y-component separately
  - the x- and y-positions of the particles over time are returned

- batch_postprocess
  run this file as batch in order to post-process all the data of the parameter identification for the settings in the paper (2,12,16)

- postprocessing.py
  run this file to generate the figures shown in the paper by using the appropriate input parameter for the initial parameter choice 

- adjoint_implicit.py
  computes the solution of the adjoint system with an implicit solver, the right-hand side of the adjoint involves the an optimal tranport plan which is computed using the python OT library
  - adjoint_interaction_force assembles the system matrix of the implicit solver
 
- cost.py
  evaluates the cost functional for given data, the Wasserstein metric is evaluated with the help of the OT library for python
  - return values are J and the values of the single terms of the cost functional

- gradient.py
  computes the gradient for given forward, adjoint and control data


To run the code you should have python 3 installed with numpy and ot library.
via the terminal start the code using. The code is run for a specific choice of input parameters eta and theta:

python main.py 2

The argument "2" chooses setting 2 in the init.py, you may choose from setting 1-16 (or define additional settings).
Choosing settings like above facilitates to run the code in parallel on a cluster. 
All required output files are saved as text files.
To visualise the output, one can run code for postprocessing for the specific setting, e.g. for setting 2:

python postprocessing.py 2

