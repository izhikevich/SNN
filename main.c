/* SNN transformer architecture in pure C. */
/* Based on Spiking Manifesto (Izhikevich 2025) */
/* The code strives for simplicity, not efficiency. */
/* Eugene Izhikevich, October 2025 */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#define FILE_NAME "loss.csv"    // saves validation loss values during training

#define CONTEXT_SIZE            32
#define VOCAB_SIZE              256
#define EMBEDDING_DIM           32
#define POSITIONAL_DIM          4     
#define NUM_LAYERS              6
#define NUM_HEADS               4

#define N_T                     16      
#define N_C                     6     


#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define sign(x) (x > 0 ? 1 : -1)        // zero has "minus" sign
#define U(x) ( 0.5/(1+fabs(x)) )        // not used; here for reference
#define Up(x) (-0.5*sign(x)/(1+fabs(x))/(1+fabs(x))) 

#define LEARNING_RATE  (MIN( 1 /sqrt(1+t), t/(4000)/sqrt(4000) )) // Adam learning rate scheduler
float learning_rate;
#define TESTING_LENGTH 10000


// ----------------------------------------------------------------------------
// Spiking model

typedef struct {
    int a[N_C];     
    int b[N_C];
} Anchors;

typedef struct {
    int y_dim;       // dimension of the output y
    float* S[N_T];   // synaptic values (table_size, y_dim), i.e., (,j,k) in the paper
    Anchors anchors[N_T];    
} LUT;

typedef struct {
    int r_min[N_T];
    float u_min[N_T];
    int j[N_T];      // j = H_i(x)
} LUTcache;

typedef struct {
    LUT V;
    LUTcache V_cache[CONTEXT_SIZE];

    float Positional_encoding[CONTEXT_SIZE][N_T][POSITIONAL_DIM];  // position -> embedding
    LUTcache PE_cache[CONTEXT_SIZE];

} AttentionHead;

typedef struct {
    float Token_embedder[VOCAB_SIZE][EMBEDDING_DIM];    // token -> embedding
    int tokens[CONTEXT_SIZE+1];                         // currrent training sample

    float z[CONTEXT_SIZE][EMBEDDING_DIM];               // resnet connections: each layer adds to z
    
    LUT FFN[NUM_LAYERS];
    LUTcache FFN_cache[NUM_LAYERS][CONTEXT_SIZE];

    AttentionHead head[NUM_LAYERS][NUM_HEADS];

    LUT unembedder;
    LUTcache unembedder_vars[CONTEXT_SIZE];

    float output[CONTEXT_SIZE][VOCAB_SIZE];             // the output becomes the error gradient in the backward pass 
} Model;



void softmax(float* x, int size, float temperature) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf( (x[i] - max_val) / temperature );
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

float vector_multiply(float* vector1, float* vector2, int size) {
    float result = 0;
    for (int i = 0; i < size; i++) {
        result += vector1[i] * vector2[i];
    }
    return result;
}

void random_vector(float* vector, int size, float scale) {
    for (int i = 0; i < size; i++) {
        vector[i] = scale * 2*( (float)rand()/RAND_MAX - 0.5);
    }
}

int sample(float* probabilities, int n) {
    // sample index from probabilities (they must sum to 1!)
    float coin = (float)rand() / RAND_MAX;
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

void fill_vector_with_random_intergers(int* vector, int N, int Max_value) {

    for (int i = 0; i < N; i++) {   
        vector[i] = rand()%Max_value;
    }
}

void fill_vector_with_random_intergers_different_from_vector2(int* vector, int* vector2, int N, int Max_value) {

    for (int i = 0; i < N; i++) {   
        do {
            vector[i] = rand()%Max_value;
        } while (vector2[i] == vector[i]);
    }
}

void build_LUT(LUT* lut, int total_n_c, int y_dim) {

    lut->y_dim = y_dim;
    for (int i = 0; i < N_T; i++) {
        fill_vector_with_random_intergers(lut->anchors[i].a, N_C, EMBEDDING_DIM);
        fill_vector_with_random_intergers_different_from_vector2(lut->anchors[i].b, lut->anchors[i].a, N_C, EMBEDDING_DIM);
        lut->S[i] = (float*)calloc(  (1 << total_n_c) * y_dim, sizeof(float));
        memset(lut->S[i], 0, (1 << total_n_c) * y_dim * sizeof(float));
    }
}

void free_LUT(LUT* lut) {

    for (int i = 0; i < N_T; i++) {
        free(lut->S[i]);
    }
}

void cache_index(LUT* lut, LUTcache* cache, float* x) {

    for (int i = 0; i < N_T; i++) { 
        cache->j[i] = 0;
        cache->u_min[i] = INFINITY;

        for (int r = 0; r < N_C; r++) { // loop over all comparisons per table
            float u = x[lut->anchors[i].a[r]] - x[lut->anchors[i].b[r]];
            if (u > 0) {
                cache->j[i] |= (1 << r); // concatenation 
            }
            if (fabs(u) < fabs(cache->u_min[i])) {
                cache->r_min[i] = r;
                cache->u_min[i] = u;
            }
        }
    }
}

void cache_PE_index(LUTcache* cache, float u[N_T][POSITIONAL_DIM]) {

    for (int i = 0; i < N_T; i++) { 
        cache->j[i] = 0;
        cache->u_min[i] = INFINITY;

        for (int r = 0; r < POSITIONAL_DIM; r++) { 
            if (u[i][r] > 0) {          // No anchor neurons; just a comparison with 0; works best for small POSITIONAL_DIM.
                cache->j[i] |= (1 << r);
            }
            if (fabs(u[i][r]) < fabs(cache->u_min[i])) {
                cache->r_min[i] = r;
                cache->u_min[i] = u[i][r];
            }
        }
    }
}

void LUT_forward(LUT* lut, LUTcache* cache, float* y) {

    // main loop in the forward pass
    for (int i = 0; i < N_T; i++) { 
        for (int k = 0; k < lut->y_dim; k++) {
            y[k] += lut->S[i][  cache->j[i] * lut->y_dim + k ];
        }
    }
}

#define BACKWARD_UPDATE(cache, gradient) \
    do { \
        float gi = 0; \
        for (int k = 0; k < lut->y_dim; k++) { \
            gi += y_gradient[k] * ( lut->S[i][ jbar + k ] - lut->S[i][ j + k ] ); \
        } \
        float v = gi * Up(cache->u_min[i]); \
        gradient[lut->anchors[i].a[cache->r_min[i]]] += v; \
        gradient[lut->anchors[i].b[cache->r_min[i]]] -= v; \
    } while (0)
//

void LUT_backward(LUT* lut, LUTcache* cache, float* x_gradient, float* y_gradient) {

    for (int i = 0; i < N_T; i++) {
        int j = cache->j[i] * lut->y_dim;
        // flip the bit number r_min 
        int jbar = ( cache->j[i] ^ (1 << cache->r_min[i]) ) * lut->y_dim;
        
        BACKWARD_UPDATE(cache, x_gradient);

        for (int k = 0; k < lut->y_dim; k++) {
            lut->S[i][ j + k] -= learning_rate * y_gradient[k];
        }
    }
}

#define CONCATENATE(Q, P, PE) ((((Q) << (N_C+POSITIONAL_DIM)) | ((P) << POSITIONAL_DIM) | (PE)) * lut->y_dim)


void concatenated_LUT_forward(LUT* lut,  LUTcache* cacheQ, LUTcache* cacheK, LUTcache* cachePE, float* y) {

    for (int i = 0; i < N_T; i++) { 
        int j = CONCATENATE( cacheQ->j[i], cacheK->j[i], cachePE->j[i] );
        for (int k = 0; k < lut->y_dim; k++) {
            y[k] += lut->S[i][ j + k ];
        }
    }
}

void concatenated_LUT_backward(LUT* lut, LUTcache* cacheQ, LUTcache* cacheK, LUTcache* cachePE, float* x_gradientQ, float* x_gradientK, float PE_grad[N_T][POSITIONAL_DIM], float* y_gradient) {

    for (int i = 0; i < N_T; i++) {

        int j = CONCATENATE( cacheQ->j[i], cacheK->j[i], cachePE->j[i] );
    
        if ( fabs(cacheQ->u_min[i]) < fabs(cacheK->u_min[i]) ) {
            int jbar  = CONCATENATE( cacheQ->j[i] ^ (1 << cacheQ->r_min[i]), cacheK->j[i], cachePE->j[i] ); // only jQ is flipped
            BACKWARD_UPDATE(cacheQ, x_gradientQ);
        }
        else {
            int jbar  = CONCATENATE( cacheQ->j[i], cacheK->j[i] ^ (1 << cacheK->r_min[i]), cachePE->j[i] ); // only jP is flipped
            BACKWARD_UPDATE(cacheK, x_gradientK);
        }

        if ( fabs(cachePE->u_min[i]) < fabs(cacheQ->u_min[i]) && fabs(cachePE->u_min[i]) < fabs(cacheK->u_min[i]) ) {
            int jbarPE = CONCATENATE( cacheQ->j[i], cacheK->j[i], cachePE->j[i] ^ (1 << cachePE->r_min[i]) ); // only jPE is flipped
            float giPE = 0;
            for (int k = 0; k < lut->y_dim; k++) {
                giPE += y_gradient[k] * ( lut->S[i][ jbarPE + k ] - lut->S[i][ j + k ] );
            }
            float deltaPE = giPE * Up(cachePE->u_min[i]);
            PE_grad[i][cachePE->r_min[i]] += deltaPE; // No anchor neurons; just a comparison with 0.
        }

        for (int k = 0; k < lut->y_dim; k++) {
            lut->S[i][ j + k] -= learning_rate * y_gradient[k];
        }
    }
}

void build_Model(Model* m) {

    random_vector(m->Token_embedder[0], VOCAB_SIZE*EMBEDDING_DIM, 1.0f);
 
    for (int l = 0; l < NUM_LAYERS; l++) {
        // the inputs are z_pos
        build_LUT(&m->FFN[l], N_C, EMBEDDING_DIM); 
        for (int h = 0; h < NUM_HEADS; h++) {
            random_vector(m->head[l][h].Positional_encoding[0][0], CONTEXT_SIZE*N_T*POSITIONAL_DIM, 1.0f);
            // the inputs are concatenated [Q, P, PE]
            build_LUT(&m->head[l][h].V, N_C + N_C + POSITIONAL_DIM, EMBEDDING_DIM); 
        }
    }
    build_LUT(&m->unembedder,  N_C, VOCAB_SIZE);
}

void free_Model(Model* m) {

    for (int l = 0; l < NUM_LAYERS; l++) {
        free_LUT(&m->FFN[l]);
        for (int h = 0; h < NUM_HEADS; h++) {
            free_LUT(&m->head[l][h].V);
        }
    }
    free_LUT(&m->unembedder);
}

void attention_forward(AttentionHead* head, float x[CONTEXT_SIZE][EMBEDDING_DIM], float y[CONTEXT_SIZE][EMBEDDING_DIM]) {

    for (int pos = 0; pos < CONTEXT_SIZE; pos++) { 
        cache_index(&head->V, &head->V_cache[pos], x[pos]);
        cache_PE_index(&head->PE_cache[pos], head->Positional_encoding[pos]);
    }
    
    for (int pos = 1; pos < CONTEXT_SIZE; pos++) {
        for (int pos1 = 0; pos1 < pos; pos1++) {
            concatenated_LUT_forward(&head->V, &head->V_cache[pos], &head->V_cache[pos1], &head->PE_cache[pos-pos1], y[pos]);
        }
    }
}

void attention_backward(AttentionHead* head, float x_grad[CONTEXT_SIZE][EMBEDDING_DIM], float y_grad[CONTEXT_SIZE][EMBEDDING_DIM]) {

    float pos_grad[CONTEXT_SIZE][N_T][POSITIONAL_DIM];
    memset(pos_grad, 0, CONTEXT_SIZE*N_T*POSITIONAL_DIM*sizeof(float));

    for (int pos = 1; pos < CONTEXT_SIZE; pos++) { 
        for (int pos1 = 0; pos1 < pos; pos1++) { 
            concatenated_LUT_backward(&head->V, &head->V_cache[pos], &head->V_cache[pos1], &head->PE_cache[pos-pos1],  x_grad[pos], x_grad[pos1], pos_grad[pos-pos1], y_grad[pos]); 
        }
    }

    for (int pos = 0; pos < CONTEXT_SIZE; pos++) {
        for (int i = 0; i < N_T; i++) {
            for (int k = 0; k < POSITIONAL_DIM; k++) {
                head->Positional_encoding[pos][i][k] -= learning_rate * pos_grad[pos][i][k];
            }
        }
    }
}

void model_forward(Model* m) {

    for (int l = 0; l < NUM_LAYERS; l++) {

        // AttentionHead from all z to all z   
        float x[CONTEXT_SIZE][EMBEDDING_DIM];
        memcpy(x, m->z, CONTEXT_SIZE*EMBEDDING_DIM*sizeof(float)); // each head will be looking at the same input
        for (int h = 0; h < NUM_HEADS; h++) {
            attention_forward(&m->head[l][h], x, m->z); // resnet connections: add to the output from the previous layer, m->z
        }
    
        // FFN from z_pos to z_pos.
        // For attention-only mode, comment the 4 lines below and 3 lines in the next function
        for (int pos = 0; pos < CONTEXT_SIZE; pos++) {
            cache_index(&m->FFN[l], &m->FFN_cache[l][pos], m->z[pos]);    // resnet connections: from m->z
            LUT_forward(&m->FFN[l], &m->FFN_cache[l][pos], m->z[pos]);    // back to m->z
        }
    }
    
    memset(m->output, 0, CONTEXT_SIZE*VOCAB_SIZE*sizeof(float));
    for (int pos = 0; pos < CONTEXT_SIZE; pos++) {
        cache_index(&m->unembedder, &m->unembedder_vars[pos], m->z[pos]);    
        LUT_forward(&m->unembedder, &m->unembedder_vars[pos], m->output[pos]); 
    }
}

void model_backward(Model* m) {

    float y_grad[CONTEXT_SIZE][EMBEDDING_DIM]; 
    float x_grad[CONTEXT_SIZE][EMBEDDING_DIM];
    memset(x_grad, 0, CONTEXT_SIZE*EMBEDDING_DIM*sizeof(float));

    for (int pos = 0; pos < CONTEXT_SIZE; pos++) {
        LUT_backward(&m->unembedder, &m->unembedder_vars[pos], x_grad[pos], m->output[pos]); 
    }

    for (int l = NUM_LAYERS-1; l >= 0; l--) {

        // FFN from z_pos to z_pos
        memcpy(y_grad, x_grad, CONTEXT_SIZE*EMBEDDING_DIM*sizeof(float)); // don't zero-out x_grad, but add to it (resnet connections)
        // For attention-only mode, comment the 3 lines below and 4 lines in the previous function
        for (int pos = 0; pos < CONTEXT_SIZE; pos++) {
            LUT_backward(&m->FFN[l], &m->FFN_cache[l][pos], x_grad[pos], y_grad[pos]);
        }

        // AttentionHead from all z to all z
        memcpy(y_grad, x_grad, CONTEXT_SIZE*EMBEDDING_DIM*sizeof(float)); // don't zero-out x_grad, but add to it (resnet connections)
        for (int h = 0; h < NUM_HEADS; h++) {
            attention_backward(&m->head[l][h], x_grad, y_grad);
        }
    }

    // no need to compute gradients for the embedder; just update the synaptic values
    for (int pos = 0; pos < CONTEXT_SIZE; pos++) {
        for (int k = 0; k < EMBEDDING_DIM; k++) {
            // does not improve performance in character prediction mode, so OK to disable
            //m->Token_embedder[ m->tokens[pos] ][k] -= learning_rate * x_grad[pos][k]; 
        }
    }
}

void model_training_step(Model* m) {

    model_forward(m);
    for (int pos = 0; pos < CONTEXT_SIZE; pos++) {
        softmax(m->output[pos], VOCAB_SIZE, 1.0f);
        m->output[pos][ m->tokens[pos+1] ] -= 1.0f; // output become a gradient
    }
    model_backward(m);
}




typedef struct {
    unsigned char* data;
    int length;
    unsigned char* reserved_for_testing;
    int testing_input_data[TESTING_LENGTH];

} TrainingData;

void free_training_data(TrainingData* training) {
    if (training->data) {
        free(training->data);
    }
    if (training->reserved_for_testing) {
        free(training->reserved_for_testing);
    }
}

void load_training_data(TrainingData* training, char* fname) {
    FILE *file = fopen(fname, "rb");
    if (!file) {
        printf("Error opening training datafile %s", fname);
        exit(1);
    }
    else {
        printf("Successfully opened training data file %s\n", fname);
    }

    // Get file size
    fseek(file, 0, SEEK_END);
    int file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    // Allocate memory for the entire file
    unsigned char* local_data = (unsigned char*)malloc(file_size);
    if (!local_data) {
        perror("Error allocating memory");
        fclose(file);
        exit(1);
    }

    // Read the entire file
    training->length = fread(local_data, 1, file_size, file);
    fclose(file);

    if (training->length != file_size) {
        printf("Warning: Expected %d bytes but read %d bytes\n", file_size, training->length);
    }

    training->data = local_data;
    training->length -= CONTEXT_SIZE + 1;

    training->reserved_for_testing = (unsigned char*)calloc(training->length, sizeof(unsigned char));
    memset(training->reserved_for_testing, 0, training->length*sizeof(unsigned char));

    for (int i = 0; i < TESTING_LENGTH; i++) {
        training->testing_input_data[i] = rand() % training->length;
        for (int j = -CONTEXT_SIZE; j <= CONTEXT_SIZE; j++) { // ensure that the testing input data is not too close to the training data
            training->reserved_for_testing[MAX(0, training->testing_input_data[i]+j)] = 1;
        }
    }
    printf("Successfully loaded training data\n");
}

int get_random_training_index(TrainingData* training) {
    
    int symbol_index;
    do {
        symbol_index = rand() % training->length;
    } while (training->reserved_for_testing[symbol_index] == 1);
    return symbol_index;
}

void embed_token(Model* m, unsigned char* text, int pos, float* input) {

    memcpy(input, m->Token_embedder[ text[pos] ], EMBEDDING_DIM*sizeof(float));
}

void load_snippet(Model* m, TrainingData* training, int char_start) {

    for (int pos = 0; pos < CONTEXT_SIZE; pos++) {
        embed_token(m, training->data + char_start, pos, m->z[pos]); 
        m->tokens[pos] = (int)training->data[ char_start+pos ]; 
    }
    m->tokens[CONTEXT_SIZE] = (int)training->data[ char_start+CONTEXT_SIZE ]; 
}

int model_inference(Model* m) {

    model_forward(m);
    softmax(m->output[CONTEXT_SIZE-1], VOCAB_SIZE, 0.4f);
    int sampled_index = sample(m->output[CONTEXT_SIZE-1], VOCAB_SIZE);
    return sampled_index;
}

void model_prompt_response(Model* m, unsigned char* prompt, int response_length) {

    unsigned char prompt_copy[CONTEXT_SIZE+1];
    strlcpy((char *)prompt_copy, (const char *)prompt, CONTEXT_SIZE+1); // Need to fix: what if the prompt is too short?
    printf("%s", prompt_copy);

    for (int i = 0; i < response_length; i++) {
        for (int pos = 0; pos < CONTEXT_SIZE; pos++) {
            embed_token(m, prompt_copy, pos, m->z[pos]);
        }
        int response = model_inference(m);
        printf("%c", (unsigned char)response);

        // shift the prompt by one character and insert response as the last character
        for (int j = 0; j < CONTEXT_SIZE-1; j++) {
            prompt_copy[j] = prompt_copy[j+1];
        }
        prompt_copy[CONTEXT_SIZE-1] = (unsigned char)response; 
    }
}




// =================================================================================
// Main function
// =================================================================================   
int main(int argc, char *argv[]) {

    TrainingData training;
    load_training_data(&training, "train_v2_drcat_02.csv");

    FILE *file_loss = fopen(FILE_NAME, "w"); fclose(file_loss);

    Model m;
    build_Model(&m);
    
    for (int t = 0; t < 100000000; t++) {

        load_snippet(&m, &training, get_random_training_index(&training));
        learning_rate = LEARNING_RATE; // Adam scheduler
        model_training_step(&m);
    
        if (t % 10000 == 0) {

            printf("...validating... "); fflush(stdout);
            float validation_loss = 0;
            
            for (int i = 0; i < TESTING_LENGTH; i++) {                
                load_snippet(&m, &training, training.testing_input_data[i]);
                model_forward(&m);
                softmax(m.output[CONTEXT_SIZE-1], VOCAB_SIZE, 1.0f);
                validation_loss += -log(m.output[CONTEXT_SIZE-1][m.tokens[CONTEXT_SIZE]]);
            }
            validation_loss /= TESTING_LENGTH;

            FILE *file_loss = fopen(FILE_NAME, "a");
            fprintf(file_loss, "%d, %f\n", t, validation_loss);
            fclose(file_loss);
    
            printf("\rt=%d,000, loss=%5.3f: ", t/1000, validation_loss);
            model_prompt_response(&m, (unsigned char*)"insert your validation prompt here ", 80);
            printf("\n"); 
        }
        printf("\rt=%d", t); fflush(stdout);
    }
    
    free_Model(&m);
    free_training_data(&training);
    return 0;
}
