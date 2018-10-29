clean:
	rm -rf *.bmat *.png *.npy *.vector *.dat *.vec invalid.data *.vtk *.bms constraint.matrix && ls

calc:
	python ./1_create_model.py
	python ./2_create_syn_data.py

inv:
	python ./3_conventional_inversion.py

case1:
	python ./4_apply_4PM.py
	python ./5_joint_inversion.py
	python ./6_plot_inv_results.py

case2:
	python ./4_apply_4PM.py "Fig2"
	python ./5_joint_inversion.py "Fig2"
	python ./6_plot_inv_results.py "Fig2"

show:
	python ./6_plot_inv_results.py
	python ./6_plot_inv_results.py "Fig2"

copy:
	cp -f "Fig1_two_columns.pdf" ../../manuscripts/1_theory_and_implementation
	cp -f "Fig2_two_columns.pdf" ../../manuscripts/1_theory_and_implementation


all: clean calc inv case1 case2 show copy
