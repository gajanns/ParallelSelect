#include <iostream>
#include <string>
#include <random>
#include <algorithm>
#include <limits>
#include <chrono>
#include <thread>
#include <execution>
#include <semaphore>



/* #region Random Generator for Test-Input */
int random_number(int min = 0, int max = INT32_MAX)
{
    thread_local std::random_device rd;
    thread_local std::mt19937 gen(rd());
    if(max < 0){
        std::cout << "FUCK" << std::flush;
    }
    return std::uniform_int_distribution<int>(min, max)(gen);
}

void fill_random_int(std::vector<int> &container)
{
    std::generate(container.begin(), container.end(), [&container]()
                  { return random_number(0, container.size()); });
}
/* #endregion */

/* #region Reference-Algorithm for Verification */
int ref_select(std::vector<int> &sequence, size_t k)
{
    std::sort(std::execution::par_unseq, sequence.begin(), sequence.end());
    return sequence[k - 1];
}
/* #endregion */

/* #region Structure for shared-memory Data-Communication*/
struct quickselect_config{
    std::vector<int>::iterator start, end, mem_start, mem_end;
    int* pivot;
    size_t* a_size,* b_size,* c_size,* next_path;
    std::counting_semaphore<12>* sem_ready,* sem_pivot,* sem_partition,* sem_recursion ;
};

struct frselect_config{
    std::vector<int>::iterator start, end, mem_start, mem_end;
    int* pivot_left,* pivot_right;
    size_t* a_size,* b_size,* c_size,* next_path;
    std::counting_semaphore<12>* sem_ready,* sem_pivot,* sem_partition,* sem_recursion ;
};

quickselect_config from_frselect_config(frselect_config &config){
    return quickselect_config{.start = config.start, .end = config.end, .mem_start = config.mem_start, .mem_end = config.mem_end, .pivot = config.pivot_left,
    .a_size = config.a_size, .b_size = config.b_size, .c_size = config.c_size, .next_path = config.next_path};
} 

frselect_config from_quickselect_config(quickselect_config &config){
    return frselect_config{.start = config.start, .end = config.end, .mem_start = config.mem_start, .mem_end = config.mem_end, .pivot_left = config.pivot,
    .pivot_right = config.pivot, .a_size = config.a_size, .b_size = config.b_size, .c_size = config.c_size, .next_path = config.next_path};
} 

enum{
    A = 1, B = 2, C = 4, FIN = 8
};
/* #endregion*/

/* #region Partition-Routine */
void partition(frselect_config &config){
    int pivot_left, pivot_right;
    pivot_left = *(config.pivot_left);
    pivot_right = *(config.pivot_right);

    std::vector<int>::iterator a_end = config.mem_start, c_start = config.mem_start + ( config.end-config.start );
    for( auto i = config.start; i != config.end; i++ ) {
        if ( *i < pivot_left ) *( a_end++ ) = *i;
        else if ( *i > pivot_right ) *( --c_start ) = *i;
    }
    std::vector<int>::iterator b_start = a_end;
    for( auto i = config.start; i != config.end && b_start != c_start; i++ ) {
        if ( *i >= pivot_left && *i <= pivot_right) *( b_start++ ) = *i;
    }

    std::copy( config.mem_start, config.mem_start + ( config.end-config.start ), config.start );
    size_t a_size = a_end - config.mem_start, b_size = c_start - a_end, c_size = config.end-config.start-a_size-b_size;
    *(config.a_size) = a_size;
    *(config.b_size) = b_size;
    *(config.c_size) = c_size;
}

void partition(quickselect_config &config){
    frselect_config fr_config = from_quickselect_config(config);
    partition(fr_config);
}
/* #endregion */


void decide_path(size_t *a_size, size_t *b_size, size_t *c_size, size_t &k, size_t *next_path){
    if ( *(a_size) >= k ) *(next_path) = A;
    else if ( *(a_size) + *(b_size) >= k ) {*(next_path) = B; k -= *(a_size);}
    else {*(next_path) = C; k -= *(a_size) + *(b_size);}
}

int quick_select_seq_no_alloc(quickselect_config &config, size_t k, bool is_unit = false){

    // Pick or Receive Pivot
    if(is_unit) {
        config.sem_ready->release(1);
        config.sem_pivot->acquire();
    }
    else *(config.pivot) = *( config.start + random_number( 0, config.end-config.start - 1 ) );

    // Partition
    partition(config);    
    if(is_unit){
        config.sem_partition->release(1);
    }

    // Create or Receive Recursion-Path
    if(is_unit){
        config.sem_recursion->acquire();
    }
    else{
        decide_path(config.a_size,config.b_size,config.c_size, k, config.next_path);
    }
    
    // Execute Recursion or Exit
    switch(*(config.next_path)) {
        case A: 
            config.end = config.start + *(config.a_size);
            return quick_select_seq_no_alloc( config, k, is_unit );
        case C:
            config.start += *(config.a_size) + *(config.b_size);
            return quick_select_seq_no_alloc( config, k, is_unit );
        default:
            return *(config.pivot);
    }
}

int quick_select_seq(std::vector<int> &s, size_t k){
    std::vector<int> mem;
    mem.resize(s.size());
    int pivot;
    size_t a_size, b_size, c_size, next_path;
    quickselect_config config = { .start = s.begin(), .end = s.end(), .mem_start = mem.begin(), .mem_end = mem.end(), .pivot = &pivot, .a_size = &a_size,
        .b_size = &b_size, .c_size = &c_size, .next_path = &next_path
    };
    return quick_select_seq_no_alloc(config, k, false);
}

int quick_select_par(std::vector<int> &s, size_t k){
    size_t aux_thread_n = std::thread::hardware_concurrency() - 1, current_k = k, next_path = A;
    std::vector<int> mem(s.size());
    std::vector<size_t> a_size(aux_thread_n), b_size(aux_thread_n), c_size(aux_thread_n), s_size(aux_thread_n);
    std::vector<quickselect_config> configs(aux_thread_n);
    int pivot;
    std::counting_semaphore<12> sem_ready(0), sem_pivot(0), sem_partition(0), sem_recursion(0);
    
    /* #region Distribute input unto units */
    std::thread aux_threads[aux_thread_n];
    size_t block_size = s.size() / aux_thread_n;
    for (size_t i = 0; i < aux_thread_n; i++) {
        configs[i] = {
            .start = s.begin() + i * block_size, .end = (i == aux_thread_n - 1 ? s.end(): s.begin() + (i+1) * block_size ),
            .mem_start = mem.begin() + i * block_size, .mem_end = (i == aux_thread_n - 1 ? mem.end(): mem.begin() + (i+1) * block_size ),
            .pivot = &pivot,
            .a_size = &( a_size[i] ), .b_size = &( b_size[i] ), .c_size = &( c_size[i] ), .next_path = &next_path,
            .sem_ready = &sem_ready, .sem_pivot = &sem_pivot, .sem_partition = &sem_partition, .sem_recursion = &sem_recursion
        };
        aux_threads[i] = std::thread(quick_select_seq_no_alloc, std::ref(configs[i]), k, true);
    }
    /* #endregion */

    /* #region Work in Parallel*/
    while( true ){
        for(size_t i = 0; i < aux_thread_n; i++) sem_ready.acquire();  
        
        // Pick Pivot              
        for(size_t i = 0; i < aux_thread_n; i++) s_size[i] = configs[i].end-configs[i].start;
        std::inclusive_scan(s_size.begin(), s_size.end(), s_size.begin());

        size_t pivot_index = random_number( 0, s_size[aux_thread_n-1] - 1 );
        for(size_t i = 0; i < aux_thread_n; i++){
            if( pivot_index < s_size[i] ){
                pivot = *( configs[i].start + pivot_index - (i == 0 ? 0:s_size[i-1]));
                break;
            }
        }

        // Broadcast Pivot
        sem_pivot.release(aux_thread_n);

        // Wait for Partition-End
        for(size_t i = 0; i < aux_thread_n; i++){
            sem_partition.acquire();
        }

        // Reduce partly Solution
        size_t a_size_sum = std::reduce(a_size.begin(), a_size.end());
        size_t b_size_sum = std::reduce(b_size.begin(), b_size.end());
        size_t c_size_sum = s_size[aux_thread_n-1] - a_size_sum - b_size_sum;
        // Decide Recursion-Path
        decide_path(&a_size_sum, &b_size_sum, &c_size_sum, current_k, &next_path);

        // Broadcast next Recursion Path
        sem_recursion.release(aux_thread_n);

        // If result found, wait for all PEs to die
        if (next_path == B){
            for (size_t i = 0; i < aux_thread_n; i++) aux_threads[i].join();
            return pivot;
        }
    }
    /* #endregion */
}

size_t calc_delta_s(size_t n){
    return std::pow(n,0.66)*std::pow(std::log(n), 0.33);
}

int calc_delta_k(size_t n){
    return std::pow(n,0.33)*std::pow(std::log(n), 0.66);
}


int floyd_rivest_seq_no_alloc(frselect_config &config, size_t k, bool is_unit = false){

    // Pick or Receive Pivot
    if(is_unit) {
        config.sem_ready->release(1);
        config.sem_pivot->acquire();
        if(*(config.next_path) == FIN) return 0; // Parallel to Sequential
    }
    else {
        size_t delta_s = calc_delta_s(config.end-config.start);
        int delta_k = calc_delta_k(config.end-config.start);

        if((size_t)(config.end-config.start) < 600){
            quickselect_config quick_config = from_frselect_config(config);
            return quick_select_seq_no_alloc(quick_config, k);
        }

        // Pick Sample
        auto sample_start = config.start + random_number(0, config.end-config.start-delta_s-1);
        auto sample_end = sample_start+delta_s;

        // Pick Two Pivots
        quickselect_config seq_config = from_frselect_config(config);
        int tmp_pivot;
        seq_config.pivot = &tmp_pivot;
        seq_config.start = sample_start;
        seq_config.end = sample_end;
        int k_ = k* delta_s / (config.end-config.start);    
        *(config.pivot_left) = quick_select_seq_no_alloc(seq_config, std::max(1,k_-delta_k));
        seq_config.start = sample_start;
        seq_config.end = sample_end;
        *(config.pivot_right) = quick_select_seq_no_alloc(seq_config, std::min((int)delta_s,k_+delta_k));
    }

    // Partition
    partition(config);    
    if(is_unit){
        config.sem_partition->release(1);
    }

    size_t current_k = k;
    // Create or Receive Recursion-Path
    if(is_unit){
        config.sem_recursion->acquire();
    }
    else{
        decide_path(config.a_size,config.b_size,config.c_size, current_k, config.next_path);
    }
    
    // Execute Recursion or Exit
    switch(*(config.next_path)) {
        case A: 
            config.end = config.start + *(config.a_size);
            break;
        case C:
            config.start += *(config.a_size) + *(config.b_size);
            break;
        default:
            if(*(config.pivot_left) == *(config.pivot_right)) return *(config.pivot_left);
            config.start += *(config.a_size);
            config.end = config.start + *(config.b_size);
    }
    return floyd_rivest_seq_no_alloc( config, current_k, is_unit );
}

int floyd_rivest_seq(std::vector<int> &s, size_t k){
    std::vector<int> mem;
    mem.resize(s.size());
    int pivot_left, pivot_right;
    size_t a_size, b_size, c_size, next_path;
    frselect_config config = { .start = s.begin(), .end = s.end(), .mem_start = mem.begin(), .mem_end = mem.end(), .pivot_left = &pivot_left, .pivot_right = &pivot_right,
    .a_size = &a_size, .b_size = &b_size, .c_size = &c_size, .next_path = &next_path
    };
    return floyd_rivest_seq_no_alloc(config, k, false);
}

int floyd_rivest_par(std::vector<int> &s, size_t k){
    size_t aux_thread_n = std::thread::hardware_concurrency() - 1, current_k = k, next_path = A;
    std::vector<int> mem(s.size());
    std::vector<size_t> a_size(aux_thread_n), b_size(aux_thread_n), c_size(aux_thread_n), s_size(aux_thread_n);
    std::vector<frselect_config> configs(aux_thread_n);
    int pivot_left, pivot_right;
    std::counting_semaphore<12> sem_ready(0), sem_pivot(0), sem_partition(0), sem_recursion(0);
    
    /* #region Distribute input unto units */
    std::thread aux_threads[aux_thread_n];
    size_t block_size = s.size() / aux_thread_n;
    for (size_t i = 0; i < aux_thread_n; i++) {
        configs[i] = {
            .start = s.begin() + i * block_size, .end = (i == aux_thread_n - 1 ? s.end(): s.begin() + (i+1) * block_size ),
            .mem_start = mem.begin() + i * block_size, .mem_end = (i == aux_thread_n - 1 ? mem.end(): mem.begin() + (i+1) * block_size ),
            .pivot_left = &pivot_left, .pivot_right = &pivot_right, .a_size = &( a_size[i] ), .b_size = &( b_size[i] ), .c_size = &( c_size[i] ), .next_path = &next_path,
            .sem_ready = &sem_ready, .sem_pivot = &sem_pivot, .sem_partition = &sem_partition, .sem_recursion = &sem_recursion
        };
        aux_threads[i] = std::thread(floyd_rivest_seq_no_alloc, std::ref(configs[i]), k, true);
    }
    /* #endregion */

    /* #region Work in Parallel*/
    while( true ){
        for(size_t i = 0; i < aux_thread_n; i++) sem_ready.acquire();  
        
        // Pick Pivot              
        for(size_t i = 0; i < aux_thread_n; i++) s_size[i] = configs[i].end-configs[i].start;
        std::inclusive_scan(s_size.begin(), s_size.end(), s_size.begin());

        size_t delta_s = calc_delta_s(s_size[aux_thread_n-1]);
        int delta_k = calc_delta_k(s_size[aux_thread_n-1]);

        if(s_size[aux_thread_n-1] < std::max(s.size()/aux_thread_n, (size_t)600)){
            for(size_t i = 0; i< aux_thread_n; i++) std::copy(configs[i].start, configs[i].end, mem.begin()+(i==0?0:s_size[i-1]));
            next_path = FIN;
            sem_pivot.release(aux_thread_n);
            for (size_t i = 0; i < aux_thread_n; i++) aux_threads[i].join();

            quickselect_config seq_config = from_frselect_config(configs[0]);
            seq_config.start = mem.begin(), seq_config.end = mem.begin()+s_size[aux_thread_n-1], seq_config.mem_start = s.begin(), seq_config.mem_end = s.end();
            return quick_select_seq_no_alloc(seq_config, current_k, false);
        }

        size_t sample_left_index = random_number(0, s_size[aux_thread_n-1]-delta_s-1);
        size_t sample_aux = 0;
        std::vector<int>::iterator sample_left;
        for(size_t i = 0; i < aux_thread_n; i++){
            if(sample_left_index < s_size[i]) {
                sample_aux = i-1;
                sample_left = configs[i].start + sample_left_index- (i==0?0:s_size[i-1]);
                break;
            }
        }

        for(size_t i = 0; i < delta_s; i++){
            mem[i] = *(sample_left++);
            if(sample_left == configs[sample_aux].end) sample_left = configs[++sample_aux].start;
        }
        int k_ = current_k* delta_s / s_size[aux_thread_n-1];
        std::sort(std::execution::par_unseq, mem.begin(), mem.begin()+delta_s);
        pivot_left = mem[std::max(0,k_-delta_k-1)];
        pivot_right = mem[std::min((int)delta_s-1,k_+ delta_k-1)];
        
        // Broadcast Pivot
        sem_pivot.release(aux_thread_n);

        // Wait for Partition-End
        for(size_t i = 0; i < aux_thread_n; i++){
            sem_partition.acquire();
        }

        // Reduce partly Solution
        size_t a_size_sum = std::reduce(a_size.begin(), a_size.end());
        size_t b_size_sum = std::reduce(b_size.begin(), b_size.end());
        size_t c_size_sum = s_size[aux_thread_n-1] - a_size_sum - b_size_sum;
        // Decide Recursion-Path
        decide_path(&a_size_sum, &b_size_sum, &c_size_sum, current_k, &next_path);
        if(next_path == B && pivot_left == pivot_right) next_path = FIN;
        // Broadcast next Recursion Path
        sem_recursion.release(aux_thread_n);

        // If result found, wait for all PEs to die
        if (next_path == FIN){
            for (size_t i = 0; i < aux_thread_n; i++) aux_threads[i].join();
            return pivot_left;
        }
    }
    /* #endregion */
}

/* #region Execute with Timer and Result-Checker */
auto perf_test_milliseconds(std::vector<int> &input, size_t k, int (*f)(std::vector<int> &, size_t), bool test = false)
{
    std::vector<int> ref_sample;
    if (test)
    {
        ref_sample.resize(input.size());
        std::copy(input.begin(), input.end(), ref_sample.begin());
    }

    auto start = std::chrono::system_clock::now();
    int result = (*f)(input, k);
    auto stop = std::chrono::system_clock::now();

    if (test && result != ref_select(ref_sample, k))
        std::cerr << "Result must be wrong." << std::endl;
    return std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
}
/* #endregion */

void extract_userinput(size_t &sample_size, size_t &sample_rep, auto &algorithm, int &argc, char *argv[]){
    if (argc != 4)
    {
        std::cerr << "Usage not correct" << std::endl;
        exit(-1);
    }

    algorithm = quick_select_seq;
    std::string algo_str(argv[1]);
    if (algo_str == "REF" || algo_str == "ref")
    {
        algorithm = ref_select;
    }
    else if (algo_str == "QSSEQ" || algo_str == "qsseq")
    {
        algorithm = quick_select_seq;
    }
    else if (algo_str == "QSPAR" || algo_str == "qspar")
    {
        algorithm = quick_select_par;
    }
    else if (algo_str == "FRSEQ" || algo_str == "frseq")
    {
        algorithm = floyd_rivest_seq;
    }
    else if (algo_str == "FRPAR" || algo_str == "frpar")
    {
        algorithm = floyd_rivest_par;
    }
    else
    {
        std::cerr << argv[1] << "Usage not correct" << std::endl;
        exit(-1);
    }
    std::string size_str(argv[2]);
    std::string rep_str(argv[3]);

    try
    {
        sample_size = (size_t)std::stod(size_str);
        sample_rep = (size_t)std::stoi(rep_str);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        exit(-1);
    }
}

int main(int argc, char *argv[]){
    auto *algorithm = floyd_rivest_par;
    size_t sample_size = 1e6, sample_rep = 5;
    //extract_userinput(sample_size, sample_rep, algorithm, argc, argv);
    std::cout<<"Size:"<<sample_size<<" Rep:"<<sample_rep<<std::flush;
    std::vector<int> sample(sample_size);
    auto time = 0;
    for (size_t i = 0; i < sample_rep; i++)
    {
        fill_random_int(sample);
        size_t k = random_number(1, sample_size);
        auto dur = perf_test_milliseconds(sample, k, algorithm, true);
        std::cout << " dur:" << dur << "ms " << std::flush;
        time += dur;
    }
    /* #endregion */

    /* #region Output */
    std::cout << std::endl
              << " Sample-Size: " << sample_size << std::endl
              << " Sample-Reps: " << sample_rep << std::endl
              << " Average-Time: " << time / (double)sample_rep << "ms" << std::endl;
    /* #endregion */
}
