// DBSCAN_serial.cpp
// Serial point classification for DBSCAN-style core/outlier detection
// Usage: DBSCAN_serial.exe input.csv eps min_samples [output.csv]
// - input.csv: comma-separated rows with at least two numeric columns (x,y,...). No header.
// - eps: float (Euclidean distance threshold). Note: we compare squared distances to avoid computing sqrt.
// - min_samples: integer >= 1 (counts other points only, not the point itself).
// - output.csv: optional. If omitted, will use input filename with "_results.csv" suffix.

#include <bits/stdc++.h>
using namespace std;

// powershell -NoProfile -ExecutionPolicy Bypass -File .\run_serial_bench.ps1

/*
Esto lo que hace es dar la posibilidad de usar el comando 
.\DBSCAN_serial.exe 120000_data.csv 0.05 10 
para que el programa agarre el archivo 120000_data.csv con 0.05 de eps y 10 de min_samples
y si no se pone el nombre del archivo de salida, lo crea automaticamente con el nombre 120000_results.csv
*/
static void print_usage(const string &prog) {
    cerr << "Usage: " << prog << " input.csv eps min_samples [output.csv]\n";
}

int main(int argc, char** argv) {
    if (argc < 4) {
        print_usage(argv[0]); // esto es para obtener el nombre del programa
        return 1;
    }

    string input_path = argv[1]; // esto saca el 2do string del comando que es el nombre del archivo
    float eps = 0.0f; // inicializamos epsilon 
    int min_samples = 0; // inicializamos min_samples
    try {
        eps = stof(argv[2]); // convertimos el string del 3er argumento a float
        min_samples = stoi(argv[3]); // convertimos el string del 4to argumento a int
    } catch (...) { // Por si no jala
        cerr << "Error: invalid eps or min_samples.\n";
        return 1;
    }
    if (eps <= 0.0f || min_samples < 1) { // validamos los valores
        cerr << "Error: eps must be > 0 and min_samples >= 1\n";
        return 1;
    }

    string output_path; // nombre del archivo de salida
    if (argc >= 5) output_path = argv[4]; // si se pone el nombre del archivo de salida, lo usamos
    else {
        // Hace un replace de _data.csv por _results.csv
        size_t pos = input_path.rfind("_data.csv"); 
        if (pos != string::npos) output_path = input_path.substr(0, pos) + "_results.csv";
        else output_path = input_path + "_results.csv";
    }

    ios::sync_with_stdio(false); 
    cin.tie(nullptr); 

    // Read input CSV
    ifstream fin(input_path); 
    if (!fin) {
        cerr << "Error: cannot open input file: " << input_path << "\n";
        return 1;
    }

    vector<vector<float>> points; // Guardamos los puntos leidos del archivo
    string line;
    size_t line_no = 0;
    while (getline(fin, line)) { 
        ++line_no;
        if (line.find_first_not_of(" \t\r\n") == string::npos) continue; // skip empty lines
        vector<float> row; // vector para guardar los valores de cada fila
        string token;
        stringstream ss(line);
        bool parse_error = false;
        while (getline(ss, token, ',')) {
            // Agarramos cada token separado por comas
            size_t a = token.find_first_not_of(" \t\r\n"); 
            size_t b = token.find_last_not_of(" \t\r\n"); 
            if (a==string::npos) { parse_error = true; break; } 
            string t = token.substr(a, b-a+1); 
            try {
                float v = stof(t); // convertimos el token a float
                row.push_back(v); // lo guardamos en el vector de la fila
            } catch (...) {
                parse_error = true;
                break;
            }
        }
        if (parse_error) {
            // per user request: skip rows if malformed
            cerr << "Warning: skipping malformed row " << line_no << "\n";
            continue;
        }
        if (row.size() < 2) {
            cerr << "Warning: skipping row " << line_no << " (need at least 2 columns)\n";
            continue;
        }
        points.push_back(move(row));
    }
    fin.close();

    size_t N = points.size(); // Cantidad de puntos validos leidos
    if (N == 0) {
        cerr << "No valid points read from input.\n";
        return 1;
    }

    size_t D = points[0].size(); // Dimensionalidad de los puntos
    for (size_t i = 1; i < N; ++i) if (points[i].size() != D) {
        cerr << "Error: inconsistent dimensionality at row " << i+1 << "\n";
        return 1;
    }

    cout << "Read " << N << " points of dimension " << D << "\n";


    // Los labels van a ser 1 si es core de primer orden y 0 si es outlier
    vector<int> label_type(N, 0); // inicializamos todos los labels en 0 (outlier)
    float eps2 = eps * eps; // Comparamos distancias al cuadrado para evitar sqrt

    // Para cada punto, contamos cuántos otros puntos están dentro de eps (excluyéndose a sí mismo)
    for (size_t i = 0; i < N; ++i) { // Recorremos todos los puntos 
        int count = 0;
        for (size_t j = 0; j < N; ++j) { // Recorremos todos los puntos para comparar 
            if (i == j) continue; // no contamos a sí mismo
            float s = 0.0f; // s es la distancia al cuadrado
            for (size_t k = 0; k < D; ++k) { // Recorremos todas las dimensiones
                float d = points[i][k] - points[j][k]; // i y j son las filas, k es la columna
                // por ejemplo: d = x_i - x_j en la primer dimensión -> s = d
                //              d = y_i - y_j en la segunda dimensión
                s += d * d; // x^2 + y^2
            }
            if (s <= eps2) { // si c^2 <= eps^2 entonces está dentro del radio
                ++count;
                if (count >= min_samples) break; // suficiente para declarar core
            }
        }
        if (count >= min_samples) label_type[i] = 1; // core
        else label_type[i] = 0; // outlier
    }
    // Paso 2
    for (size_t i = 0; i < N; ++i) {
        if (label_type[i] != 0) continue; // Solo con los outliers
        for (size_t j = 0; j < N; ++j) {
            if (label_type[j] != 1) continue; // Solo con los cores
            float s = 0.0f;
            for (size_t k = 0; k < D; ++k) {
                float d = points[i][k] - points[j][k];
                s += d * d;
            }
            if (s <= eps2) { label_type[i] = 1; break; }
        }
    }


    // Escribe el CSV: x,y,type
    ofstream fout(output_path); 
    if (!fout) {
        cerr << "Error: cannot open output file: " << output_path << "\n";
        return 1;
    }
    fout.setf(std::ios::fixed); fout<<setprecision(6);
    for (size_t i = 0; i < N; ++i) {
        fout << points[i][0] << "," << points[i][1] << "," << label_type[i] << "\n";
    }
    fout.close();

    cout << "Wrote results to " << output_path << " (cores=" << accumulate(label_type.begin(), label_type.end(), 0) << ")\n";
    return 0;
}
