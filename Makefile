all: acor

acor: force
	python setup.py build_ext --inplace

clean:
	rm -rf build acor/*.pyc acor/_acor.so

force:
	true
