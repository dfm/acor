all: acor

acor: acor/acor.c acor/acor.h acor/_acor.c
	python setup.py build_ext --inplace

clean:
	rm -rf build acor/*.pyc acor/_acor.so

