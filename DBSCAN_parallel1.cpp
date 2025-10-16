// DBSCAN_parallel1.cpp
// Parallel (OpenMP) implementation for first-order and second-order core detection
// Two modes:
// 1) Single-run: DBSCAN_parallel1.exe input.csv eps min_samples threads [output.csv]
// 2) Benchmark: DBSCAN_parallel1.exe --bench eps min_samples iterations
//    - uses datasets {20000,40000,80000,120000} from current dir (files named <N>_data.csv)
//    - threads tested: {1, vcores/2, vcores, vcores*2}
//    - does 1 warm-up then 'iterations' measured runs per configuration

#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

/*
Para hacer pruebas usar el comando:
.\DBSCAN_parallel1.exe 20000_data.csv 0.03 10 4

Para correr el benchmark 
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
        if (bad) continue; // skip malformed
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

// Write output CSV with first two columns and type
bool write_results_xy_type(const string &path, const vector<float> &data, size_t N, size_t D, const vector<int> &type) {
    ofstream fout(path);
    if (!fout) return false;
    fout.setf(std::ios::fixed); fout<<setprecision(6);
    for (size_t i = 0; i < N; ++i) {
        fout << data[i*D + 0] << "," << data[i*D + 1] << "," << type[i] << "\n";
    }
    return true;
}

// Single-run compute (Phase A and Phase B)
void compute_labels_parallel(const vector<float> &data, size_t N, size_t D, float eps, int min_samples, vector<int> &label_type, int nthreads, int chunk) {
    float eps2 = eps * eps;
    label_type.assign(N, 0);

    omp_set_num_threads(nthreads);
    // Phase A: first-order cores
    #pragma omp parallel for schedule(dynamic, chunk)
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

    // Phase B: second-order (outlier that is within eps of any first-order core)
    #pragma omp parallel for schedule(dynamic, chunk)
    for (size_t i = 0; i < N; ++i) {
        if (label_type[i] != 0) continue; // only previously outliers
        for (size_t j = 0; j < N; ++j) {
            if (label_type[j] != 1) continue; // only first-order cores
            float s = 0.0f;
            for (size_t d = 0; d < D; ++d) {
                float diff = data[i*D + d] - data[j*D + d];
                s += diff * diff;
            }
            if (s <= eps2) { label_type[i] = 2; break; }
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
        vector<int> datasets = {20000, 40000, 80000, 120000};

        unsigned vcores = thread::hardware_concurrency();
        if (vcores == 0) vcores = 1;
        vector<int> thread_list;
        thread_list.push_back(1);
        thread_list.push_back(max(1u, vcores/2));
        thread_list.push_back((int)vcores);
        thread_list.push_back((int)min<unsigned>(vcores*2, vcores*2)); // allow vcores*2

        // Open output CSVs
        ofstream raw_out("bench_raw.csv");
        ofstream summary_out("bench_summary.csv");
        raw_out << "dataset,n_points,threads,iteration,elapsed_ms,cores1,cores2,total_cores\n";
        summary_out << "dataset,n_points,threads,mean_ms,stddev_ms,mean_cores1,mean_cores2\n";

        for (int n_points : datasets) {
            string input = to_string(n_points) + "_data.csv";
            vector<float> data;
            size_t N=0,D=0;
            if (!read_csv_flat(input, data, N, D)) {
                cerr << "Warning: cannot read " << input << "; skipping dataset\n";
                continue;
            }
            cout << "Dataset "<< n_points << ": read " << N << " points (D="<<D<<")\n";

            for (int threads : thread_list) {
                cout << "Running: N="<<N<<" threads="<<threads<<" iterations="<<iterations<<"\n";
                // Warm-up
                vector<int> labels;
                compute_labels_parallel(data, N, D, eps, min_samples, labels, threads, 1);

                vector<double> times_ms;
                vector<int> cores1_list, cores2_list;
                for (int it = 0; it < iterations; ++it) {
                    labels.assign(N, 0);
                    double t0 = omp_get_wtime();
                    compute_labels_parallel(data, N, D, eps, min_samples, labels, threads, 1);
                    double t1 = omp_get_wtime();
                    double elapsed_ms = (t1 - t0) * 1000.0;
                    int c1 = 0, c2 = 0;
                    for (size_t i = 0; i < N; ++i) {
                        if (labels[i] == 1) ++c1;
                        else if (labels[i] == 2) ++c2;
                    }
                    times_ms.push_back(elapsed_ms);
                    cores1_list.push_back(c1);
                    cores2_list.push_back(c2);
                    raw_out << "dataset" << "," << N << "," << threads << "," << it << "," << elapsed_ms << "," << c1 << "," << c2 << "," << (c1+c2) << "\n";
                    raw_out.flush();
                    cout << "  it="<<it<<" ms="<<elapsed_ms<<" cores1="<<c1<<" cores2="<<c2<<"\n";
                }

                // compute mean/stddev
                double sum=0; for (double v: times_ms) sum+=v; double mean = sum / times_ms.size();
                double s=0; for (double v: times_ms) s += (v-mean)*(v-mean); double stddev = sqrt(s / max(1,(int)times_ms.size()));
                double mean_c1=0, mean_c2=0; for (int v: cores1_list) mean_c1+=v; for (int v: cores2_list) mean_c2+=v; mean_c1/=cores1_list.size(); mean_c2/=cores2_list.size();
                summary_out << "dataset" << "," << N << "," << threads << "," << mean << "," << stddev << "," << mean_c1 << "," << mean_c2 << "\n";
                summary_out.flush();
            }
        }

        raw_out.close(); summary_out.close();
        cout << "Benchmark finished. Raw: bench_raw.csv, Summary: bench_summary.csv\n";
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
    if (!read_csv_flat(input, data, N, D)) { cerr<<"Error reading input file\n"; return 1; }
    vector<int> labels;
    double t0 = omp_get_wtime();
    compute_labels_parallel(data, N, D, eps, min_samples, labels, threads, 1);
    double t1 = omp_get_wtime();
    int c1=0,c2=0; for (size_t i=0;i<N;++i){ if (labels[i]==1) ++c1; else if (labels[i]==2) ++c2; }
    cout << "Done. N="<<N<<" D="<<D<<" threads="<<threads<<" time_ms="<< (t1-t0)*1000.0 <<" cores1="<<c1<<" cores2="<<c2<<"\n";
    if (!write_results_xy_type(output, data, N, D, labels)) cerr<<"Warning: cannot write output file\n";
    else cout << "Wrote "<< output <<"\n";

    return 0;
}
