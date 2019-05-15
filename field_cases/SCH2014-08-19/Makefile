all: prepare mesh inv joint all

prepare:
	bash 1_get_and_prepare_data.sh

mesh:
	python3 2_make_mesh_and_filter.py

inv:
	python3 3_conventional_inversion.py && python3 4_apply_4PM.py

joint:
	python3 5_joint_inversion.py

plot:
	python3 6_plot_inv_results.py
