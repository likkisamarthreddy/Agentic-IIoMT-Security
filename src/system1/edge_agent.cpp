#include <iostream>
#include <vector>
#include <deque>
#include <string>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <mosquitto.h>
#include <onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>
#include <arpa/inet.h>
#include <net/if.h>

using json = nlohmann::json;

// Global configuration
std::string agent_id = "edge-1";
std::string mqtt_host = "mosquitto";
int mqtt_port = 1883;
std::string traffic_topic;
std::string alert_topic;
std::string cmd_topic;

int sequence_length = 20; // default, can be overridden by env
int num_features = 46; // default, can be overridden by env
const double ANOMALY_THRESHOLD = 0.85; // Simplified KDE threshold

std::deque<std::vector<float>> packet_buffer;
long long total_packets = 0;
long long anomalies_detected = 0;

// ONNX Runtime globals
Ort::Env* ort_env = nullptr;
Ort::Session* ort_session = nullptr;
Ort::MemoryInfo* memory_info = nullptr;

void apply_sdn_mitigation(const std::string& action, const std::string& src_ip) {
    if (action.find("THROTTLE") != std::string::npos || action.find("DROP") != std::string::npos) {
        std::cout << "[EMERGENCY BRAKE] Using standard Linux tc to THROTTLE IP " << src_ip << "..." << std::endl;
        system("tc qdisc replace dev eth0 root tbf rate 1mbit burst 32kbit latency 400ms");
    }
}

void forward_alert(struct mosquitto* mosq, const json& alert_data) {
    json payload = alert_data;
    payload["source_agent"] = agent_id;
    payload["timestamp"] = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
        
    std::string payload_str = payload.dump();
    mosquitto_publish(mosq, NULL, alert_topic.c_str(), payload_str.length(), payload_str.c_str(), 0, false);
}

void process_packet(struct mosquitto* mosq, const json& packet_data) {
    total_packets++;
    
    if (packet_data.contains("features") && packet_data["features"].is_array()) {
        std::vector<float> features;
        for (auto& f : packet_data["features"]) {
            features.push_back(f.get<float>());
        }
        
        if (features.size() == num_features) {
            packet_buffer.push_back(features);
            if (packet_buffer.size() > sequence_length) {
                packet_buffer.pop_front();
            }
        }
    }
    
    if (packet_buffer.size() < sequence_length) {
        return; // Need full sequence
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Prepare input tensor [1, sequence_length, num_features]
    std::vector<float> input_tensor_values(sequence_length * num_features);
    for (int i = 0; i < sequence_length; ++i) {
        for (int j = 0; j < num_features; ++j) {
            input_tensor_values[i * num_features + j] = packet_buffer[i][j];
        }
    }
    
    std::vector<int64_t> input_shape = {1, sequence_length, num_features};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        *memory_info, input_tensor_values.data(), input_tensor_values.size(),
        input_shape.data(), input_shape.size()
    );
    
    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};
    
    auto output_tensors = ort_session->Run(
        Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1
    );
    
    float* floatarr = output_tensors.front().GetTensorMutableData<float>();
    // Simplified anomaly scoring:
    float max_logit = floatarr[0];
    for (int i = 1; i < 6; ++i) {
        if (floatarr[i] > max_logit) max_logit = floatarr[i];
    }
    float sum_exp = 0.0f;
    for (int i = 0; i < 6; ++i) {
        sum_exp += std::exp(floatarr[i] - max_logit);
    }
    float max_prob = std::exp(max_logit - max_logit) / sum_exp;
    float anomaly_score = 1.0f - max_prob;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double latency_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    std::cout << "Inference Latency: " << latency_ms << " ms | Anomaly Score: " << anomaly_score << std::endl;
    
    bool is_anomalous = anomaly_score > ANOMALY_THRESHOLD;
    
    if (is_anomalous) {
        anomalies_detected++;
        
        json result = {
            {"packet_id", total_packets},
            {"device_id", packet_data.value("device_id", "unknown")},
            {"score", anomaly_score},
            {"is_anomalous", true},
            {"latency_ms", latency_ms},
            {"brake_triggered", false}
        };
        
        // Emergency Brake
        if (anomaly_score > 0.95) {
            result["brake_triggered"] = true;
            result["brake_action"] = "DROP";
            std::string src_ip = packet_data.value("src_ip", "10.0.0.99");
            apply_sdn_mitigation("DROP", src_ip);
        }
        
        forward_alert(mosq, result);
    }
}

void on_message(struct mosquitto* mosq, void* userdata, const struct mosquitto_message* msg) {
    if (!msg->payload) return;
    
    std::string topic(msg->topic);
    std::string payload_str((char*)msg->payload, msg->payloadlen);
    
    try {
        json data = json::parse(payload_str);
        if (topic == traffic_topic) {
            process_packet(mosq, data);
        } else if (topic == cmd_topic) {
            std::cout << "Received gateway command: " << payload_str << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error parsing MQTT message: " << e.what() << std::endl;
    }
}

int main(int argc, char* argv[]) {
    if (const char* env_id = std::getenv("AGENT_ID")) agent_id = env_id;
    if (const char* env_host = std::getenv("MQTT_HOST")) mqtt_host = env_host;
    if (const char* env_port = std::getenv("MQTT_PORT")) mqtt_port = std::stoi(env_port);
    if (const char* env_features = std::getenv("NUM_FEATURES")) num_features = std::stoi(env_features);
    if (const char* env_seq = std::getenv("SEQUENCE_LENGTH")) sequence_length = std::stoi(env_seq);
    
    traffic_topic = "iimt/traffic/" + agent_id;
    alert_topic = "iimt/edge/alerts";
    cmd_topic = "iimt/gateway/commands";
    
    std::cout << "Initializing Edge Agent: " << agent_id << std::endl;
    
    // Init ONNX Runtime
    ort_env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "EdgeAgent");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    
    try {
        ort_session = new Ort::Session(*ort_env, "checkpoints/cnn_bigru_int8.onnx", session_options);
        memory_info = new Ort::MemoryInfo(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));
        std::cout << "Loaded ONNX INT8 model successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error loading ONNX model: " << e.what() << std::endl;
        std::cerr << "Ensure checkpoints/cnn_bigru_int8.onnx exists!" << std::endl;
    }
    
    // Init Mosquitto
    mosquitto_lib_init();
    struct mosquitto* mosq = mosquitto_new(agent_id.c_str(), true, NULL);
    if (!mosq) {
        std::cerr << "Error initializing Mosquitto." << std::endl;
        return 1;
    }
    
    mosquitto_message_callback_set(mosq, on_message);
    
    if (mosquitto_connect(mosq, mqtt_host.c_str(), mqtt_port, 60) != MOSQ_ERR_SUCCESS) {
        std::cerr << "Could not connect to MQTT broker." << std::endl;
        return 1;
    }
    
    std::cout << "Connected to MQTT broker at " << mqtt_host << ":" << mqtt_port << std::endl;
    
    mosquitto_subscribe(mosq, NULL, traffic_topic.c_str(), 0);
    mosquitto_subscribe(mosq, NULL, cmd_topic.c_str(), 0);
    
    mosquitto_loop_forever(mosq, -1, 1);
    
    // Cleanup
    mosquitto_destroy(mosq);
    mosquitto_lib_cleanup();
    delete ort_session;
    delete ort_env;
    delete memory_info;
    
    return 0;
}
