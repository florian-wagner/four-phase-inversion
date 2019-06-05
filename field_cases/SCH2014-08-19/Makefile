all: prepare mesh inv joint1 joint2 plot

prepare:
	bash 1_get_and_prepare_data.sh

mesh:
	python3 2_make_mesh_and_filter.py

inv:
	python3 3_conventional_inversion.py && python3 4_apply_4PM.py

joint1:
	python3 5_joint_inversion.py 1

joint2:
	python3 5_joint_inversion.py 2

plot:
	python3 6_plot_inv_results.py
