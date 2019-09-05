clean:
	rm -rf *.bmat *.npz *.png *.npy *.vector *.dat *.vec invalid.data *.vtk *.bms constraint.matrix && ls

calc:
	python3 ./1_create_model.py
	python3 ./2_create_syn_data.py

inv:
	python3 ./3_conventional_inversion.py 1
	python3 ./3_conventional_inversion.py 2

case1:
	python3 ./4_apply_4PM.py
	python3 ./5_joint_inversion.py
	python3 ./6_plot_inv_results.py

case2:
	python3 ./4_apply_4PM.py "Fig2"
	python3 ./5_joint_inversion.py "Fig2"
	python3 ./6_plot_inv_results.py "Fig2"

show:
	python3 ./6_plot_inv_results.py
	python3 ./6_plot_inv_results.py "Fig2"
	xdg-open "Fig1_two_columns.pdf" & xdg-open "Fig2_two_columns.pdf"
	python3 8_plot_matrix.py

matrix:
	python3 7_calc_model_covariance.py

all: clean calc inv case1 case2 matrix show
