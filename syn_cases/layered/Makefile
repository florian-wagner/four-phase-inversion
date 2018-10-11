clean:
	rm -rf *.bmat *.png *.npy *.npz *.vector *.dat *.vec invalid.data *.vtk *.bms constraint.matrix && ls

calc:
	python ./1_create_model.py
	python ./2_create_syn_data.py

inv1:
	python ./3_conventional_inversion.py
	python ./4_joint_inversion.py

inv2:
	python ./3_conventional_inversion.py "Fig2"
	python ./4_joint_inversion.py "Fig2"

show:
	python ./5_plot_results.py
	python ./5_plot_results.py "Fig2"

copy:
	cp -f "Fig1_two_columns.pdf" ../../manuscripts/1_theory_and_implementation
	cp -f "Fig2_two_columns.pdf" ../../manuscripts/1_theory_and_implementation


all: clean calc inv1 inv2 show copy
