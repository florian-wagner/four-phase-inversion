all: prepare mesh inv joint1 joint2 show

prepare:
	bash 1_copy_data.sh

mesh:
	python3 2_make_mesh_and_filter.py

inv:
	python3 3_conventional_inversion.py && python3 4_apply_4PM.py

joint1:
	python3 5_joint_inversion.py 1

joint2:
	python3 5_joint_inversion.py 2

show:
	python3 6_plot_inv_results.py
	python3 7_plot_boreholes.py

clean:
	rm -rf *.bmat *.npz *.png *.npy *.vector *.dat *.vec invalid.data *.vtk *.bms constraint.matrix && ls
