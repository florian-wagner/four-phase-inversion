clean:
	rm -rf *.bmat *.png *.npy *.npz *.vector *.dat *.vec invalid.data *.vtk *.bms constraint.matrix && ls

calc:
	python ./1_create_model.py
	python ./2_create_syn_data.py

inv:
	python ./3_conventional_inversion.py
	python ./4_joint_inversion.py

show:
	python ./5_plot_results.py

all: clean calc inv show
