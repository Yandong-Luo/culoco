#include <iostream>
#include <vector>
#include <Eigen/Dense>

// Pinocchio 头文件
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/spatial/explog.hpp"

// 使用 Eigen 命名空间
using namespace Eigen;

// 一个用于储存您提供的张量输入的类
class TensorInput {
public:
    // 矩阵存储所有关节角度配置
    MatrixXd data;
    
    // 构造函数
    TensorInput(const std::vector<std::vector<double>>& input_data) {
        int rows = input_data.size();
        int cols = input_data[0].size();
        
        data = MatrixXd(rows, cols);
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data(i, j) = input_data[i][j];
            }
        }
    }
    
    // 获取指定行
    VectorXd getRow(int row_index) const {
        return data.row(row_index);
    }
    
    // 获取行数
    int getNumRows() const {
        return data.rows();
    }
    
    // 获取列数
    int getNumCols() const {
        return data.cols();
    }
};

// 前向运动学函数 - 处理单个构型
pinocchio::SE3 forwardKinematics(const pinocchio::Model& model, pinocchio::Data& data, const VectorXd& q, int end_effector_id) {
    // 确保输入向量的长度正确
    assert(q.size() >= model.nq && "输入向量长度必须大于或等于模型中的关节数");
    
    // 使用向量前model.nq个元素进行前向运动学计算
    VectorXd q_model = q.head(model.nq);
    
    // 计算前向运动学
    pinocchio::forwardKinematics(model, data, q_model);
    
    // 更新所有框架的位置 - 需要传入配置q
    pinocchio::framesForwardKinematics(model, data, q_model);
    
    // 返回末端执行器的位姿
    return data.oMf[end_effector_id];
}

// 前向运动学函数 - 处理批量构型
std::vector<pinocchio::SE3> forwardKinematicsBatch(const pinocchio::Model& model, pinocchio::Data& data, 
                                        const TensorInput& tensor_input, 
                                        int end_effector_id) {
    std::vector<pinocchio::SE3> results;
    
    // 对每一行执行前向运动学
    for (int i = 0; i < tensor_input.getNumRows(); i++) {
        VectorXd row = tensor_input.getRow(i);
        
        // 确保行的长度为12（您的URDF有12个关节）
        if (row.size() < model.nq) {
            std::cerr << "警告: 行 " << i << " 只有 " << row.size() 
                     << " 个元素，但模型需要 " << model.nq << " 个" << std::endl;
            continue;
        }
        
        // 使用单个构型的前向运动学函数
        pinocchio::SE3 pose = forwardKinematics(model, data, row, end_effector_id);
        results.push_back(pose);
    }
    
    return results;
}

// 逆向运动学函数
bool inverseKinematics(const pinocchio::Model& model, pinocchio::Data& data, 
                       const pinocchio::SE3& target_pose, VectorXd& q_result,
                       int end_effector_id,
                       double eps = 1e-4, int max_iter = 1000, 
                       double dt = 1e-1, double damp = 1e-6) {
    
    // 创建雅可比矩阵和误差向量
    MatrixXd J(6, model.nv);
    VectorXd error(6);
    VectorXd v(model.nv);
    
    // 迭代求解
    for (int i = 0; i < max_iter; i++) {
        // 更新前向运动学
        pinocchio::forwardKinematics(model, data, q_result);
        pinocchio::framesForwardKinematics(model, data, q_result);
        
        // 计算当前位姿与目标位姿之间的误差
        const pinocchio::SE3& current_pose = data.oMf[end_effector_id];
        const pinocchio::SE3 error_se3 = current_pose.inverse() * target_pose;
        
        // 将位姿误差转换为空间误差
        error.head<3>() = error_se3.translation();
        error.tail<3>() = pinocchio::log3(error_se3.rotation());
        
        // 检查是否达到精度要求
        if (error.norm() < eps) {
            return true;
        }
        
        // 计算当前构型的雅可比矩阵
        pinocchio::computeFrameJacobian(model, data, q_result, end_effector_id, 
                                       pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J);
        
        // 使用阻尼伪逆方法计算关节速度
        MatrixXd JJt = J * J.transpose();
        for (int k = 0; k < 6; k++) {
            JJt(k, k) += damp;
        }
        v = J.transpose() * JJt.ldlt().solve(error);
        
        // 更新构型
        q_result = pinocchio::integrate(model, q_result, dt * v);
        
        // 可选：应用关节限制
        for (int j = 0; j < model.nq; j++) {
            if (q_result[j] < model.lowerPositionLimit[j]) {
                q_result[j] = model.lowerPositionLimit[j];
            } else if (q_result[j] > model.upperPositionLimit[j]) {
                q_result[j] = model.upperPositionLimit[j];
            }
        }
    }
    
    // 如果达到最大迭代次数仍未收敛，返回失败
    return false;
}

int main(int argc, char** argv) {
    // 检查是否提供了URDF文件路径
    if (argc < 2) {
        std::cerr << "用法: " << argv[0] << " 路径/到/机器人.urdf [末端执行器关节名称]" << std::endl;
        return 1;
    }
    
    std::string urdf_filename = argv[1];
    std::string end_effector_name = (argc > 2) ? argv[2] : "tool0"; // 默认末端执行器关节名称
    
    // 创建Pinocchio模型和数据
    pinocchio::Model model;
    pinocchio::Data data(model);
    
    // 加载URDF模型
    try {
        // 加载URDF模型，verbose = true用于调试
        pinocchio::urdf::buildModel(urdf_filename, model);
        data = pinocchio::Data(model);
        
        std::cout << "模型加载成功，有 " << model.nq << " 个位置自由度和 " 
                  << model.nv << " 个速度自由度" << std::endl;
                  
        // 检查模型是否有12个关节
        if (model.nq != 12) {
            std::cout << "警告: 您的模型有 " << model.nq 
                     << " 个关节，但您提到应该有12个关节" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "加载模型错误: " << e.what() << std::endl;
        return 1;
    }
    
    // 确定末端执行器关节ID
    int end_effector_id;
    try {
        // 获取末端执行器的框架ID
        end_effector_id = model.getFrameId(end_effector_name);
        std::cout << "末端执行器 '" << end_effector_name << "' 找到，ID: " << end_effector_id << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "寻找末端执行器错误: " << e.what() << std::endl;
        std::cout << "可用框架:" << std::endl;
        for (size_t i = 0; i < model.frames.size(); ++i) {
            std::cout << "- " << model.frames[i].name << " (ID: " << i << ")" << std::endl;
        }
        return 1;
    }
    
    // ======== 使用您提供的输入张量 ========
    
    // 创建输入张量 - 我们只使用第一行作为示例，因为您说您的URDF有12个关节
    // 其他行可以用于测试不同的构型
    std::vector<std::vector<double>> tensor_data = {
        {0.5435, 0.2722, 0.2846, 0.4873, 0.5020, 0.5468, 0.8945, 0.4650, 0.4464, 0.4764, 0.0172, 0.4585},
        {0.3162, 0.7049, 0.6659, 0.4427, 0.4371, 0.3229, 0.4327, 0.4718, 0.8529, 0.0186, 0.9995, 0.3997},
        {0.9265, 0.7983, 0.3808, 0.3579, 0.8486, 0.4648, 0.9982, 0.6311, 0.4139, 0.3773, 0.8135, 0.1818},
        {0.4821, 0.0166, 0.9257, 0.3280, 0.7627, 0.8781, 0.9519, 0.3795, 0.3818, 0.6929, 0.5701, 0.9751},
        {0.8543, 0.9227, 0.1659, 0.4146, 0.5440, 0.0731, 0.1580, 0.8505, 0.9666, 0.2864, 0.1540, 0.5065},
        {0.6276, 0.9457, 0.6927, 0.4801, 0.7937, 0.4502, 0.2447, 0.0510, 0.1033, 0.5689, 0.4113, 0.2120},
        {0.5519, 0.3928, 0.2161, 0.2607, 0.7517, 0.4555, 0.0207, 0.7148, 0.3986, 0.1482, 0.3000, 0.4588},
        {0.0342, 0.1728, 0.4264, 0.7528, 0.6443, 0.6710, 0.8265, 0.0126, 0.3033, 0.9752, 0.9094, 0.5003},
        {0.2598, 0.1554, 0.3221, 0.9108, 0.0185, 0.4434, 0.8556, 0.5974, 0.2195, 0.2543, 0.3372, 0.2386},
        {0.5476, 0.5863, 0.7186, 0.5998, 0.2344, 0.9758, 0.3443, 0.6594, 0.9993, 0.0245, 0.3015, 0.1799},
        {0.1416, 0.7988, 0.9530, 0.9235, 0.0272, 0.4528, 0.8201, 0.5914, 0.6683, 0.2802, 0.0103, 0.5579},
        {0.3815, 0.8533, 0.3237, 0.9873, 0.9823, 0.2002, 0.0027, 0.3898, 0.0940, 0.8465, 0.1081, 0.3611}
    };
    
    TensorInput tensor_input(tensor_data);
    
    std::cout << "\n--- 前向运动学 ---" << std::endl;
    std::cout << "输入张量形状: [" << tensor_input.getNumRows() << ", " 
              << tensor_input.getNumCols() << "]" << std::endl;
    
    // 执行前向运动学 - 对所有提供的行
    std::vector<pinocchio::SE3> poses = forwardKinematicsBatch(model, data, tensor_input, end_effector_id);
    
    // 打印所有构型的结果
    std::cout << "\n所有构型的前向运动学结果:" << std::endl;
    for (int i = 0; i < (int)poses.size(); i++) {
        std::cout << "\n构型 " << i+1 << ":" << std::endl;
        std::cout << "关节值: " << tensor_input.getRow(i).transpose() << std::endl;
        std::cout << "末端执行器位置: " << poses[i].translation().transpose() << std::endl;
        // 可选: 打印旋转矩阵
        //std::cout << "末端执行器旋转矩阵:\n" << poses[i].rotation() << std::endl;
    }
    
    std::cout << "\n--- 逆向运动学 ---" << std::endl;
    
    // 选择第一个构型的位姿作为目标
    pinocchio::SE3 target_pose = poses[0];
    
    // 稍微修改目标位姿以演示IK
    target_pose.translation() += Vector3d(0.05, 0.05, 0.05);  // 在x、y、z方向上各移动5cm
    
    std::cout << "目标位置: " << target_pose.translation().transpose() << std::endl;
    
    // 从第二个构型开始尝试
    VectorXd q_init = tensor_input.getRow(1);
    
    // 执行逆向运动学
    bool success = inverseKinematics(model, data, target_pose, q_init, end_effector_id);
    
    if (success) {
        std::cout << "IK收敛成功" << std::endl;
    } else {
        std::cout << "IK未能收敛，尝试最大迭代次数后" << std::endl;
    }
    
    std::cout << "结果构型: " << q_init.transpose() << std::endl;
    
    // 验证结果
    pinocchio::forwardKinematics(model, data, q_init);
    pinocchio::framesForwardKinematics(model, data, q_init);
    const pinocchio::SE3& result_pose = data.oMf[end_effector_id];
    
    std::cout << "得到的位置: " << result_pose.translation().transpose() << std::endl;
    std::cout << "位置误差: " << (target_pose.translation() - result_pose.translation()).norm() << std::endl;
    
    // 尝试不同的起始配置进行逆运动学
    std::cout << "\n尝试其他起始点:" << std::endl;
    for (int j = 2; j < std::min(5, (int)tensor_input.getNumRows()); j++) {
        VectorXd q_test = tensor_input.getRow(j);
        bool ik_success = inverseKinematics(model, data, target_pose, q_test, end_effector_id);
        
        std::cout << "\n起始构型 " << j+1 << ":" << std::endl;
        std::cout << "IK结果: " << (ik_success ? "成功" : "失败") << std::endl;
        
        if (ik_success) {
            // 验证结果
            pinocchio::forwardKinematics(model, data, q_test);
            pinocchio::framesForwardKinematics(model, data, q_test);
            const pinocchio::SE3& test_pose = data.oMf[end_effector_id];
            
            std::cout << "位置误差: " << (target_pose.translation() - test_pose.translation()).norm() << std::endl;
        }
    }
    
    return 0;
}