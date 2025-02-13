// PyEmbed.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <Python.h>
#include <filesystem>
#include <iostream>
#include <cstdlib>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <limits.h>
#endif

// 获取可执行文件所在目录
std::filesystem::path get_exe_path()
{
#ifdef _WIN32
    wchar_t buffer[MAX_PATH];
    GetModuleFileNameW(nullptr, buffer, MAX_PATH);
    return std::filesystem::path(buffer).parent_path();
#else
    char buffer[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (len != -1)
    {
        buffer[len] = '\0';
        return std::filesystem::path(buffer).parent_path();
    }
    return "";
#endif
}

int main()
{
    // 1. 清除所有可能的环境变量
    _putenv("PYTHONPATH=");
    _putenv("PYTHONHOME=");
    _putenv("CONDA_PREFIX=");

    // 2. 初始化配置
    PyConfig config;
    PyConfig_InitPythonConfig(&config);

    // 3. 设置绝对隔离
    config.isolated = 1;       // 最高隔离级别
    config.use_environment = 0; // 完全忽略环境变量
    config.configure_c_stdio = 0; // 防止标准流重定向

    // 4. 强制设置路径
    std::filesystem::path exe_dir = get_exe_path();
    std::wstring python_home = exe_dir.wstring();
    PyConfig_SetString(&config, &config.home, python_home.c_str());

    // 正确设置模块搜索路径的代码
    const wchar_t* new_paths[] = {
        (exe_dir / "DLLs").wstring().c_str(),
        (exe_dir / "Lib").wstring().c_str(),
        exe_dir.wstring().c_str()
    };
    Py_ssize_t path_count = sizeof(new_paths) / sizeof(new_paths[0]);

    // 正确调用方式
    PyStatus status = PyConfig_SetWideStringList(
        &config,
        &config.module_search_paths,
        path_count,  // 路径数量（Py_ssize_t 类型）
        const_cast<wchar_t**>(new_paths)  // 路径数组
    );
    
    // 6. 初始化 Python 解释器
    status = Py_InitializeFromConfig(&config);
    if (PyStatus_Exception(status))
    {
        std::cerr << "Python 初始化失败: " << status.err_msg << std::endl;
        PyConfig_Clear(&config);
        return 1;
    }    

    // 8. 导入模块并调用函数
    PyObject* pModule = PyImport_ImportModule("image2cad");
    if (!pModule)
    {
        PyErr_Print();
        Py_Finalize();
        PyConfig_Clear(&config);
        return 1;
    }

    // 获取 png_to_svg 函数
    PyObject* pFunc = PyObject_GetAttrString(pModule, "png_to_svg");
    if (!pFunc || !PyCallable_Check(pFunc))
    {
        std::cerr << "无法找到 png_to_svg 函数或它不是可调用对象！" << std::endl;
        Py_XDECREF(pFunc);
        Py_XDECREF(pModule);
        Py_Finalize();
        PyConfig_Clear(&config);
        return 1;
    }

    // 准备参数（文件路径）
    const char* file_path = R"(D:/Image2CADPy/TestData/pdfImages/page_28_image_1.png)";  // 使用原始字符串避免转义问题
    PyObject* pArgs = PyTuple_New(1);
    PyTuple_SetItem(pArgs, 0, PyUnicode_FromString(file_path));

    // 调用 png_to_svg 函数
    PyObject* pResult = PyObject_CallObject(pFunc, pArgs);
    if (!pResult)
    {
        std::cerr << "调用 png_to_svg 函数失败！" << std::endl;
        PyErr_Print();  // 打印 Python 错误信息
    }
    else
    {
        std::cout << "png_to_svg 函数调用成功！" << std::endl;
        Py_XDECREF(pResult);
    }

    // 清理
    Py_XDECREF(pFunc);
    Py_XDECREF(pModule);
    Py_Finalize();
    PyConfig_Clear(&config);

    return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
