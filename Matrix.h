#pragma once

/*This file includes the helper matrix class that will increase the efficency of calculations*/
/*Note that this is not a fully inclusive matrix class, it only contains the functions needed for this application*/

#include <iostream>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cassert>
#include <algorithm>
#include <functional>

class Matrix {
private:
    int rows;
    int columns;
    std::vector<std::vector<double>> data;

public:
    // Default constructor
    Matrix() : rows(0), columns(0) {}

    // Constructor with dimensions
    Matrix(int rows, int columns) : rows(rows), columns(columns) {
        data.resize(rows, std::vector<double>(columns));
    }

    // Copy constructor
    Matrix(const Matrix& other) : rows(other.rows), columns(other.columns), data(other.data) {}


    int getColumns() { return columns; }
    int getRows() { return rows; }
    double getData(int r, int c) const { return data[r][c]; }
    void setData(int r, int c, double val) { data[r][c] = val; }

    // Element access
    double& operator()(int row, int col) { return data[row][col]; }

    void resize(int numRows, int numColumns)
    {
        // Create a new temporary matrix with the desired size
        Matrix temp(rows, columns);

        // Determine the number of rows and columns to copy
        int minRows = std::min(rows, numRows);
        int minColumns = std::min(columns, numColumns);

        // Copy the elements from the original matrix to the temporary matrix
        for (int i = 0; i < minRows; i++)
            for (int j = 0; j < minColumns; j++)
                temp.data[i][j] = data[i][j];

        // Update the matrix dimensions
        numRows = rows;
        numColumns = columns;

        // Copy the elements back from the temporary matrix to the resized matrix
        data = temp.data;
    }

    // Addition operator
    Matrix operator+(const Matrix& operand) const {
        Matrix result(rows, columns);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < columns; ++j) {
                result(i, j) = data[i][j] + operand.getData(i, j);
            }
        }
        return result;
    }

    // Calculates the sum of the entire matrix
    double sum() const
    {
        double sum = 0.0;
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < columns; ++j)
            {
                sum += data[i][j];
            }
        }
        return sum;
    }

    // Subtraction operator
    Matrix operator-(const Matrix& operand) const {
        Matrix result(rows, columns);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < columns; ++j) {
                result(i, j) = data[i][j] - operand.getData(i, j);
            }
        }
        return result;
    }

    // Multiplication operator
    Matrix operator*(const Matrix& operand) const {
        assert(columns == operand.rows);

        int resultRows = rows;
        int resultCols = operand.columns;
        Matrix result(resultRows, resultCols);
        for (int i = 0; i < resultRows; ++i) {
            for (int j = 0; j < resultCols; ++j) {
                int acc = 0;

                for (int k = 0; k < columns; ++k) {
                    acc += data[i][k] * operand.getData(k, j);
                }
                result.setData(i, j, acc);
            }
        }
        return result;
    }

    // Multiplication by constant
    Matrix operator*(double constant) const {
        Matrix result(rows, columns);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < columns; ++j) {
                result(i, j) = data[i][j] * constant;
            }
        }
        return result;
    }

    // Element multiplication
    Matrix multiplyElementwise(const Matrix& other) const {
        assert(rows == other.rows && columns == other.columns);

        Matrix result(rows, columns);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < columns; ++j) {
                double element = data[i][j] * other.data[i][j];
                result.setData(i, j, element);
            }
        }

        return result;
    }

    //Transpose
    Matrix transpose() const {
        Matrix transposed(columns, rows);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < columns; ++j) {
                transposed(j, i) = data[i][j];
            }
        }
        return transposed;
    }

    // Sigmoid Transformation (squishes value between 0 and 1)
    Matrix sigmoidTransformation()
    {
        Matrix result(rows, columns);
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < columns; ++j)
            {
                result(i, j) = 1.0 + exp(-data[i][j]);
            }
        }
        return result;
    }

    // Display matrix
    void display() const {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < columns; ++j) {
                std::cout << data[i][j] << "\t";
            }
            std::cout << std::endl;
        }
    }

    void allocate(int r, int c) {
        rows = r;
        columns = c;
        data.resize(rows, std::vector<double>(columns, 0.0));
    }

    // Read matrix from file
    void readFromFile(std::ifstream& file)
    {
        int rows, cols;
        file >> rows >> cols;

        allocate(rows, cols);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                double value;
                file >> value;
                setData(i, j, value);
            }
        }
    }

    // Write matrix to file
    void writeToFile(std::ofstream& file) const
    {
        file << rows << " " << columns << std::endl;

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < columns; ++j) {
                file << data[i][j] << " ";
            }
            file << std::endl;
        }
    }
};
