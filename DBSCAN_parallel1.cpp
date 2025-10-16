#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

/*

g++ -std=c++17 -O2 -fopenmp -o DBSCAN_parallel1.exe DBSCAN_parallel1.cpp

Para hacer pruebas usar el comando:
los parametros son: nombre del archivo, eps, min_samples, threads y nombre del archivo de salida (opcional)
.\DBSCAN_parallel1.exe 20000_data.csv 0.03 10 4

Para correr el benchmark 
los parametros son: --bench, eps, min_samples, iterations
.\DBSCAN_parallel1.exe --bench 0.03 10 10
*/

static void print_usage(const string &prog) {
    cerr << "Usage:\n";
    cerr << "  Single run: " << prog << " input.csv eps min_samples threads [output.csv]\n";
    cerr << "  Benchmark:  " << prog << " --bench eps min_samples iterations\n";
}

// Esto es para leer el CSV y guardarlo en un vector plano (N x D), saltando filas que no se puedan parsear
bool read_csv_flat(const string &path, vector<float> &data, size_t &N, size_t &D) {
    ifstream fin(path); // abre el archivo
    if (!fin) return false; 
    string line;
    vector<vector<float>> rows; // guarda las filas temporalmente
    while (getline(fin, line)) {
        if (line.find_first_not_of(" \t\r\n") == string::npos) continue; // skip empty lines
        string token;
        stringstream ss(line);
        vector<float> row; // vector para guardar los valores de cada fila
        bool bad = false;
        while (getline(ss, token, ',')) {
            size_t a = token.find_first_not_of(" \t\r\n");
            if (a == string::npos) { bad = true; break; }
            size_t b = token.find_last_not_of(" \t\r\n");
            string t = token.substr(a, b-a+1);
            try { row.push_back(stof(t)); } catch(...) { bad = true; break; }
        }
        if (bad) continue; // skip malos
        if (row.size() < 2) continue;
        rows.push_back(move(row));
    }
    fin.close();
    if (rows.empty()) return false;
    N = rows.size(); 
    D = rows[0].size();
    data.resize(N * D);
    for (size_t i = 0; i < N; ++i) {
        if (rows[i].size() != D) return false;
        for (size_t d = 0; d < D; ++d) data[i*D + d] = rows[i][d];
    }
    return true;
}

// Escribe el CSV: x,y,type
bool write_results_xy_type(const string &path, const vector<float> &data, size_t N, size_t D, const vector<int> &type) {
    ofstream fout(path);
    if (!fout) return false;
    fout.setf(std::ios::fixed); fout<<setprecision(6);
    for (size_t i = 0; i < N; ++i) {
        fout << data[i*D + 0] << "," << data[i*D + 1] << "," << type[i] << "\n";
    }
    return true;
}

// Calcula los labels en paralelo, EMPIEZA EL PROCESO DBSCAN :D
/*
data: vector de puntos
N: cantidad de puntos
D: dimensionalidad
eps: radio
min_samples: cantidad minima de puntos para ser core
label_type: vector de labels a llenar (0=outlier, 1=core1)
nthreads: cantidad de threads a usar
chunk: chunk size para schedule dynamic
*/
void compute_labels_parallel(const vector<float> &data, size_t N, size_t D, float eps, int min_samples, vector<int> &label_type, int nthreads, int chunk) { 
    float eps2 = eps * eps;
    label_type.assign(N, 0);

    omp_set_num_threads(nthreads);
    // Phase A: first-order cores
    #pragma omp parallel for schedule(dynamic, chunk)
    // Uso schedule dynamic para balancear carga con chunks = 1 para que 
    // cuando uno termine se eche el siguiente luego luego

    // Mismo proceso que en el serial
    // Paso 1
    for (size_t i = 0; i < N; ++i) {
        int count = 0;
        for (size_t j = 0; j < N; ++j) {
            if (i == j) continue;
            float s = 0.0f;
            for (size_t d = 0; d < D; ++d) {
                float diff = data[i*D + d] - data[j*D + d];
                s += diff * diff;
            }
            if (s <= eps2) {
                ++count;
                if (count >= min_samples) break;
            }
        }
        if (count >= min_samples) label_type[i] = 1; else label_type[i] = 0;
    }

    // Paso 2
    #pragma omp parallel for schedule(dynamic, chunk)
    for (size_t i = 0; i < N; ++i) {
        if (label_type[i] != 0) continue; // solo outliers
        for (size_t j = 0; j < N; ++j) {
            if (label_type[j] != 1) continue; // solo cores
            float s = 0.0f;
            for (size_t d = 0; d < D; ++d) {
                float diff = data[i*D + d] - data[j*D + d];
                s += diff * diff;
            }
            if (s <= eps2) { label_type[i] = 1; break; }
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) { print_usage(argv[0]); return 1; }

    string arg1 = argv[1];
    if (arg1 == "--bench") { 
        if (argc < 5) { print_usage(argv[0]); return 1; } 
        float eps = stof(argv[2]); 
        int min_samples = stoi(argv[3]); 
        int iterations = stoi(argv[4]); 

        vector<int> datasets = {20000, 40000, 80000, 120000, 200000};

        vector<int> thread_list = {1, 10, 20, 40}; 

        // Open output CSVs
        ofstream raw_out("parallel1_raw.csv");
        ofstream summary_out("parallel1_summary.csv");
        raw_out << "n_points,threads,iteration,elapsed_ms,cores\n";
        summary_out << "n_points,threads,mean_ms,cores\n";

        for (int n_points : datasets) {
            string input = to_string(n_points) + "_data.csv";
            vector<float> data;
            size_t N=0,D=0;
            if (!read_csv_flat(input, data, N, D)) {
                cerr << "No se puede abrir " << input << "; saltandolo\n";
                continue;
            }
            cout << "Dataset "<< n_points <<" _data.csv\n";

            for (int threads : thread_list) {
                cout << "Corriendo: N="<<N<<" threads="<<threads<<" con="<<iterations<<" iteraciones \n";
                // Warm-up
                vector<int> labels;
                compute_labels_parallel(data, N, D, eps, min_samples, labels, threads, 1);

                vector<double> times_ms;
                vector<int> cores_list;
                for (int it = 0; it < iterations; ++it) {
                    labels.assign(N, 0);
                    double t0 = omp_get_wtime();
                    compute_labels_parallel(data, N, D, eps, min_samples, labels, threads, 1);
                    double t1 = omp_get_wtime();
                    double elapsed_ms = (t1 - t0) * 1000.0;
                    int c = 0;
                    for (size_t i = 0; i < N; ++i) {
                        if (labels[i] == 1) ++c;
                    }
                    times_ms.push_back(elapsed_ms);
                    cores_list.push_back(c);
                    raw_out << N << "," << threads << "," << (it+1) << "," << elapsed_ms << "," << c << "\n";
                    raw_out.flush();
                    cout << "  it="<<it<<" ms="<<elapsed_ms<<" cores="<<c<<"\n";
                }

                // compute mean
                double sum=0; 
                for (double v: times_ms) sum+=v; 
                double mean = sum / times_ms.size();
                summary_out << N << "," << threads << "," << mean << "," << cores_list[0] << "\n";
                summary_out.flush(); // el flush es para asegurarse que se escriba vonforme va corriendo
            }
        }

        raw_out.close(); summary_out.close();
        cout << "Revisa parallel1_raw.csv y parallel1_summary.csv\n";
        return 0;
    }

    // Single run mode
    if (argc < 5) { print_usage(argv[0]); return 1; }
    string input = argv[1];
    float eps = stof(argv[2]);
    int min_samples = stoi(argv[3]);
    int threads = stoi(argv[4]);
    string output;
    if (argc >= 6) output = argv[5]; else {
        size_t pos = input.rfind("_data.csv"); 
        if (pos != string::npos) output = input.substr(0,pos) + "_results.csv"; else output = input + "_results.csv";
    }

    vector<float> data; size_t N=0,D=0;
    if (!read_csv_flat(input, data, N, D)) { cerr<<"Error leyendo\n"; return 1; }
    vector<int> labels;
    double t0 = omp_get_wtime();
    compute_labels_parallel(data, N, D, eps, min_samples, labels, threads, 1);
    double t1 = omp_get_wtime();
    int c=0; for (size_t i=0;i<N;++i){ if (labels[i]==1) ++c; }

    cout << "N="<<N<<" D="<<D<<" threads="<<threads<<" time_ms="<< (t1-t0)*1000.0 <<" cores="<<c<<"\n";
    if (!write_results_xy_type(output, data, N, D, labels)) cerr<<"Warning: cannot write output file\n";
    else cout << "Wrote "<< output <<"\n";

    return 0;
}
