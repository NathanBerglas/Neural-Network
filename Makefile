EXEC = bin/neuralNetwork.exe
CC = clang
DEF = "-DNODEF"
OPT = -O0
CFLAGS = -fdiagnostics-color=always -fansi-escape-codes -g -std=c17 -Wall -Werror -I include/ -MMD ${OPT} ${DEF}
VFLAGS = --log-file="results/valgrindOut.txt" --leak-check=full --show-leak-kinds=all --track-origins=yes
LIB = $(wildcard lib/*.o)
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
	valgrind ${VFLAGS} ./${EXEC} < tests/simple.in

valgrindT: ${EXEC}
	@if [ -z "$(TEST)" ]; then \
		echo "Usage: make valgrindT TEST=<test_name>"; \
	fi
	valgrind ${VFLAGS} ./${EXEC} < tests/$(TEST).in

.PHONY: run clean test

clean:
		rm -f bin/*.o bin/*.d ${EXEC}