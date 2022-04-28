all: build exec

build: svm.cpp
	g++ -o svm.exe svm.cpp -D LOCAL -O3

exec: svm.exe
	@./svm.exe