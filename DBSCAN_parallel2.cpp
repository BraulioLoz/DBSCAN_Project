#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

/*

g++ -std=c++17 -O2 -fopenmp -o DBSCAN_parallel2.exe DBSCAN_parallel2.cpp

Para hacer pruebas usar el comando:
los parametros son: nombre del archivo, eps, min_samples, threads y nombre del archivo de salida (opcional)
.\DBSCAN_parallel2.exe 20000_data.csv 0.03 10 4

Para correr el benchmark 
los parametros son: --bench, eps, min_samples, iterations
.\DBSCAN_parallel2.exe --bench 0.03 10 10
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


/*
Esto lo que hace es particionar el espacio 2D en bloques, y 
cada bloque tiene un area interior (sin halo) y un area expandida (con halo). 
Agrego los Halos para que los puntos cercanos a los bordes de los bloques puedan 
ser considerados en el calculo de distancias y no se pierdan cores.
*/
struct BBox {
    double minx, miny, maxx, maxy; // el rango en 2D
    BBox() { // constructor
        minx = miny = numeric_limits<double>::infinity(); 
        maxx = maxy = -numeric_limits<double>::infinity(); 
    } 
    // Expande el bbox para incluir el punto (x,y)
    void expand(const double x, const double y) {
        minx = min(minx, x); miny = min(miny, y);
        maxx = max(maxx, x); maxy = max(maxy, y);
    }
    // Expande el bbox en todas las direcciones por eps
    void expand_eps(double eps) {
        minx -= eps; miny -= eps; maxx += eps; maxy += eps;
    }
    // Checa si el punto (x,y) está dentro del bbox
    bool contains(double x, double y) const {
        return x >= minx && x <= maxx && y >= miny && y <= maxy;
    }
};

struct BlockInfo {
    BBox interior; // la parte del BLOQUE que seria la partición
    BBox expanded; // la partición incluyendo el overlapping
    vector<size_t> local_indices; // vector de los índices de puntos con el overlapping (halo+interior)
    vector<size_t> owned_indices; // vector de los índices de puntos sin el overlapping (interior nada más)
};

// helper: crea el bbox global que contiene todos los puntos
BBox compute_global_bbox(const vector<float> &data, size_t N, size_t D) {
    BBox b;
    for (size_t i = 0; i < N; ++i) b.expand(data[i*D + 0], data[i*D + 1]);
    return b;
}

// Construye el grid de tamaño (nx * ny) para particionar el interior en bloques
// el nx y ny son la cantidad de bloques en x y en y
vector<BlockInfo> build_block_grid(const BBox &global, int nx, int ny) {
    vector<BlockInfo> blocks; // vector para guardar los bloques
    blocks.resize(nx * ny); // resize al número de bloques
    double wx = (global.maxx - global.minx) / nx; // ancho de cada bloque
    double wy = (global.maxy - global.miny) / ny; // alto de cada bloque
    // Estos for's produucen los bloques y les asignan su bbox interior
    for (int iy = 0; iy < ny; ++iy) { 
        for (int ix = 0; ix < nx; ++ix) {
            int bi = iy * nx + ix;
            blocks[bi].interior.minx = global.minx + ix * wx; 
            blocks[bi].interior.maxx = global.minx + (ix+1) * wx;
            blocks[bi].interior.miny = global.miny + iy * wy;
            blocks[bi].interior.maxy = global.miny + (iy+1) * wy;
        }
    }
    return blocks;
}

// assign each point to all blocks whose expanded bbox will contain 
// Le da los puntos que tendría cada bloque con el overlapping 
void assign_points_to_blocks(const vector<float> &data, size_t N, size_t D, vector<BlockInfo> &blocks, double eps) {
    int B = (int)blocks.size();
    // Primero expanded blocks
    for (auto &b : blocks) {
        b.expanded = b.interior;
        b.expanded.expand_eps(eps);
        b.local_indices.clear();
        b.owned_indices.clear();
    }
    // Ahora asigna los puntos a los bloques
    for (size_t i = 0; i < N; ++i) {
        double x = data[i*D + 0];
        double y = data[i*D + 1];
        for (int bi = 0; bi < B; ++bi) {
            if (blocks[bi].expanded.contains(x,y)) { // si el punto está en el bbox expandido
                blocks[bi].local_indices.push_back(i); // lo agrega a los locales
                if (blocks[bi].interior.contains(x,y)) blocks[bi].owned_indices.push_back(i); // si está en el interior, lo agrega a los owned
            }
        }
    }
}

// Le da los labels a los puntos owned (0=outlier, 1=core). Usa solo los puntos locales para contar vecinos
// Escribe los labels en out_labels para los puntos owned
// Los puntos owned son los que están en el interior del bloque (sin halo) 
void compute_block_labels(const vector<float> &data, size_t N, size_t D, const BlockInfo &block, double eps, int min_samples, vector<int> &out_labels) {
    float eps2 = eps * eps;
    const vector<size_t> &local = block.local_indices;
    const vector<size_t> &owned = block.owned_indices;
    size_t L = local.size();
    // Si no hay puntos owned o no hay puntos locales, no hace nada
    if (L == 0 || owned.empty()) return;

    // Mapea el índice local a la posición
    unordered_map<size_t,int> idx2pos; idx2pos.reserve(L*2+1);
    for (int i = 0; i < (int)L; ++i) idx2pos[local[i]] = i;

    // Create a local array of points coordinates for faster access
    vector<float> loc_points(L * D);
    for (size_t i = 0; i < L; ++i) {
        size_t gi = local[i];
        for (size_t d = 0; d < D; ++d) loc_points[i*D + d] = data[gi*D + d];
    }

    // Paso 1: Saca los local cores (Label 1)
    vector<char> is_core_local(L, 0);
    for (size_t ii = 0; ii < L; ++ii) {
        size_t gi = local[ii];
        int count = 0;
        for (size_t jj = 0; jj < L; ++jj) {
            if (ii == jj) continue;
            float s = 0.0f;
            for (size_t d = 0; d < D; ++d) {
                float diff = loc_points[ii*D + d] - loc_points[jj*D + d];
                s += diff * diff;
            }
            if (s <= eps2) {
                ++count;
                if (count >= min_samples) break;
            }
        }
        if (count >= min_samples) is_core_local[ii] = 1;
    }

    // Paso 2: para los puntos owned, si no es core, checa si está dentro de eps de algún core local
    for (size_t oi = 0; oi < owned.size(); ++oi) {
        size_t gi = owned[oi];
        int pos = idx2pos.at(gi);
        if (is_core_local[pos]) {
            out_labels[gi] = 1; // primer orden core
            continue;
        }
        // sino, checa contra cualquier core local
        bool promoted = false;
        for (size_t jj = 0; jj < L; ++jj) {
            if (!is_core_local[jj]) continue;
            float s = 0.0f;
            for (size_t d = 0; d < D; ++d) {
                float diff = loc_points[pos*D + d] - loc_points[jj*D + d];
                s += diff * diff;
            }
            if (s <= eps2) { promoted = true; break; }
        }
        if (promoted) out_labels[gi] = 1; else out_labels[gi] = 0;
    }
}

// Particiona los puntos en bloques y corre cada bloque en paralelo
void compute_labels_halo(const vector<float> &data, size_t N, size_t D, double eps, int min_samples, vector<int> &labels, int nthreads) {
    labels.assign(N, 0);
    if (N == 0) return; 

    // Calcula el bbox global
    BBox global = compute_global_bbox(data, N, D);
    // Se elige el tamaño de la cuadrícula en función de N y los hilos. 
    int vcores = nthreads;
    int target_blocks = vcores * 4;  // target blocks = 4 veces los núcleos virtuales
    // Elegí 4 veces los núcleos virtuales para tener más bloques y mejor balance de carga
    // elijo nx, ny como sqrt(target_blocks)
    int nx = (int)round(sqrt((double)target_blocks));
    int ny = target_blocks / nx; 

    vector<BlockInfo> blocks = build_block_grid(global, nx, ny);

    // le da los puntos que tendría cada bloque con el overlapping 
    assign_points_to_blocks(data, N, D, blocks, eps);

    // Ahora corre cada bloque en paralelo: calcula los labels locales y escribe los labels owned en el array global 'labels'
    // labels locales son los que tienen el overlapping y los owned son los que no tienen el overlapping
    omp_set_num_threads(nthreads);

    #pragma omp parallel for schedule(dynamic)
    for (int bi = 0; bi < (int)blocks.size(); ++bi) {
        // cada hilo calcula un mapa temporal de out_labels para los puntos owned
        vector<int> out_labels_local(N, -1); // -1 es no visitado
        compute_block_labels(data, N, D, blocks[bi], eps, min_samples, out_labels_local);

        // escribe los owned resultados de vuelta al array global de labels
        // solo escribimos para los índices que fueron seteados (0 o 1). Condiciones de carrera: ¿múltiples bloques pueden poseer el mismo punto? Por diseño, las áreas owned son disjuntas, así que es seguro.
        for (size_t idx = 0; idx < N; ++idx) {
            if (out_labels_local[idx] != -1) labels[idx] = out_labels_local[idx];
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) { print_usage(argv[0]); return 1; }

    string arg1 = argv[1];
    if (arg1 == "--bench") {
        if (argc < 5) { print_usage(argv[0]); return 1; }
        double eps = stod(argv[2]);
        int min_samples = stoi(argv[3]);
        int iterations = stoi(argv[4]);

        vector<int> datasets = {20000, 40000, 80000, 120000, 200000};

        int vcores = omp_get_num_procs();
        vector<int> thread_list = {1, vcores/2, vcores, vcores*2};

        ofstream raw_out("parallel2_raw.csv");
        ofstream summary_out("parallel2_summary.csv");
        raw_out << "n_points,threads,iteration,elapsed_ms,cores\n";
        summary_out << "n_points,threads,mean_ms,mean_cores\n";

        for (int n_points : datasets) {
            string input = to_string(n_points) + "_data.csv";
            vector<float> data;
            size_t N=0,D=0;
            if (!read_csv_flat(input, data, N, D)) { cerr << "Cannot open "<<input<<"; skipping\n"; continue; }
            cout << "Dataset "<<n_points<<" N="<<N<<" D="<<D<<"\n";

            for (int threads : thread_list) {
                cout << "Corriendo N="<<N<<" threads="<<threads<<" con "<<iterations<<" iteraciones \n";
                // Warm-up
                vector<int> labels;
                compute_labels_halo(data, N, D, eps, min_samples, labels, threads);

                vector<double> times_ms;
                vector<int> cores_list;
                for (int it = 0; it < iterations; ++it) {
                    labels.assign(N, 0);
                    double t0 = omp_get_wtime();
                    compute_labels_halo(data, N, D, eps, min_samples, labels, threads);
                    double t1 = omp_get_wtime();
                    double elapsed_ms = (t1 - t0) * 1000.0;
                    int c = 0; for (size_t i=0;i<N;++i) if (labels[i]==1) ++c;
                    times_ms.push_back(elapsed_ms); cores_list.push_back(c);
                    raw_out << N << "," << threads << "," << (it+1) << "," << elapsed_ms << "," << c << "\n";
                    raw_out.flush();
                    cout << "  it="<<it<<" ms="<<elapsed_ms<<" cores="<<c<<"\n";
                }
                double sum=0; for (double v: times_ms) sum+=v; double mean = sum / times_ms.size();
                summary_out << N << "," << threads << "," << mean << "," << cores_list[0] << "\n";
                summary_out.flush();
            }
        }
        raw_out.close(); summary_out.close();
        cout << "Checa parallel2_raw.csv y parallel2_summary.csv\n";
        return 0;
    }

    // single run
    if (argc < 5) { print_usage(argv[0]); return 1; }
    string input = argv[1];
    double eps = stod(argv[2]);
    int min_samples = stoi(argv[3]);
    int threads = stoi(argv[4]);
    string output;
    if (argc >= 6) output = argv[5]; else {
        size_t pos = input.rfind("_data.csv");
        if (pos != string::npos) output = input.substr(0,pos) + "_results.csv"; else output = input + "_results.csv";
    }

    vector<float> data; size_t N=0,D=0;
    if (!read_csv_flat(input, data, N, D)) { cerr<<"Error reading input\n"; return 1; }
    vector<int> labels;
    double t0 = omp_get_wtime();
    compute_labels_halo(data, N, D, eps, min_samples, labels, threads);
    double t1 = omp_get_wtime();
    int c=0; for (size_t i=0;i<N;++i) if (labels[i]==1) ++c;
    cout << "N="<<N<<" D="<<D<<" threads="<<threads<<" time_ms="<< (t1-t0)*1000.0 <<" cores="<<c<<"\n";
    if (!write_results_xy_type(output, data, N, D, labels)) cerr<<"Warning: cannot write output file\n";
    else cout << "Wrote "<< output <<"\n";
    return 0;
}
