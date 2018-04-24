clean:
	rm -rf *.bmat *.npy *.npz *.vector *.dat *.vec invalid.data *.vtk *.bms constraint.matrix && ls
all:
	python3 conventional_approach.py
	python3 joint_inversion.py
	python3 plot.py
	xdg-open 4PM_joint_inversion.png
