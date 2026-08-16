================================
OPTIMIZATION CODES FOR ANALYZING LAMINATES WITH 48 PLIES
================================

Authors: José Humberto S. Almeida Jr., Saullo G. P. Castro, Emilia Balonek
Contact: (?)
Date: (?)
Affiliation: Department of Mechanical Engineering, LUT University; Department of Aerospace Structures and Materials, Delft University of Technology
Paper Title: 'Beyond Double-Double Theory: n-Directional Stacking Sequence Optimisation in Composite Laminates'
Dependencies: numpy, scipy, pymoo, composites, bfscplate2d

--------------------------------
1. OVERVIEW
--------------------------------
One code that was created for optimization of 48 plies laminate has four versions.
Each version was used to perform optimization for varying number of Double-Double (DD) laminates:
a) D_Case1 – code for n-D layout which allows one D
b) DD_Case1 – code for n-D layout which allows two D's
c) DDD_Case1 – code for n-D layout which allows three D's
d) DDDD_Case1 – code for n-D layout which allows four D's

Then, each code from these mentioned above has its own four version depending on the type
of loading chosen and whether the optimisation needs to be performed using Genetic Alghorithm (GA) or not:
i) pymoo - code considered biaxial loading and was performed using stiffness matching approach
ii) pymoo_free - code considered biaxial loading and was performed using GA
iii) pymoo_load - code considered uniaxial loading and was performed using stiffness matching approach
iv) pymoo_uniaxial_free - code considered uniaxial loading and was performed using GA

--------------------------------
2. DESCRIPTION
--------------------------------
This file describes the codes used to generate the results in the paper.
Here, the Python codes perform stacking sequence optimization of composite laminates using a Double-Double (DD) laminate configuration. 
The objective is to maximize structural performance under different types of loading and obtain buckling and failure loads.

--------------------------------
3. KEY FEATURES
--------------------------------
- Five configurations of plate dimensions 'a' and 'b' were tested giving following aspect ratios: 0.5, 1, 2, 3, and 4.  
- Biaxial in-plane loading: 'Nxx = 1.0 lb/in', 'Nyy = 0.5 lb/in' or uniaxial in-plane loading: 'Nxx = 1.0 lb/in', 'Nyy = 0 lb/in'
- Stiffness matching optimization or Genetic Algorithm (GA) approach
- Constraints set are the same througt all files: symmetric and balanced laminate

--------------------------------
4. KEY OPERATIONS 
--------------------------------
a) 'objective' function is based on stiffness matching to generate n-D layups with equivalent in-plane stiffness to 
reference laminates under biaxial and uniaxial compressive loading

b) 'objective_free' function is based on GA and explores the optimum stacking sequences that maximise 
buckling resistance and delay failure onset

--------------------------------
5. OUTPUTS
--------------------------------
- Optimal stacking sequence (in degrees)
- 'λ_cb' buckling load (in Newtons)
- 'λ_cs' failure load (in Newtons)

--------------------------------
6. RUNNING PROCEDURE OF EACH CODE
--------------------------------
1. Open environment with Python 3.11

2. Install required packages by using "pip install numpy scipy pymoo composites bfscplate2d" command

3. Define parameters such as aspect ratio

4. Run the script

5. Review output 

--------------------------------
7. REFERENCE
--------------------------------
Haftka, R. T. (1993). Optimization of laminate stacking sequence for buckling or strength. AIAA Journal, 31(5), 921–922.

--------------------------------
8. NOTES
--------------------------------
- All experiments were run on Windows 10.
- These codes were tested with Python 3.11 version.
- These codes are for research purposes and were developed for this paper.

