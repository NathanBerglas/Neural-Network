EXEC = bin/neuralNetwork.exe
CC = clang
DEF = "-DNODEF"
OPT = -O0
CFLAGS = -fdiagnostics-color=always -fansi-escape-codes -g -std=c17 -Wall -Werror -I include/ -MMD ${OPT} ${DEF}
VFLAGS = --log-file="results/valgrindOut.txt" --leak-check=full --show-leak-kinds=all --track-origins=yes
LIB = $(wildcard lib/*.o) -lm
SRC = $(wildcard src/*.c)

OBJECTS = $(SRC:src/%.c=bin/%.o)

DEPENDS = $(OBJECTS:.o=.d)
TEST_FILES := $(wildcard tests/*.in)

${EXEC}: ${OBJECTS}
		${CC} ${CFLAGS} ${OBJECTS} ${LIB} ${DEF} -o ${EXEC}

bin/%.o: src/%.c
	${CC} ${CFLAGS} -c $< -o $@

# copy the generated .d files which provides dependencies for each .c file
-include ${DEPENDS}

run: ${EXEC}
	./${EXEC}

test: ${EXEC}
	@for test_file in $(TEST_FILES); do \
		test_name=$$(basename $$test_file .in); \
		./${EXEC} < $$test_file > results/$$test_name.trace; \
	done

valgrind: ${EXEC}
	@if [ -z "$(TEST)" ]; then \
		echo "Usage: make valgrind TEST=<test_name>"; \
	fi
	valgrind ${VFLAGS} ./${EXEC} < tests/$(TEST).in

convert: bin/txtToBin.o
	./bin/txtToBin.o $(FILE)

.PHONY: run clean test sanitize valgrind convert

clean:
		rm -f bin/*.o bin/*.d ${EXEC}

# other tools not included:
#./bin/txtToBin stem
#	Converts any txt training data to binary to be used by neural network training
#gcc src/*.c -I include/ -g -Wall -Werror -fsanitize=address -lm -o bin/gccNeuralNetwork.exe
#	Compiles for address sanitizer to help fix memory errors