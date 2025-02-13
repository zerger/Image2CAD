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

// ��ȡ��ִ���ļ�����Ŀ¼
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
    // 1. ������п��ܵĻ�������
    _putenv("PYTHONPATH=");
    _putenv("PYTHONHOME=");
    _putenv("CONDA_PREFIX=");

    // 2. ��ʼ������
    PyConfig config;
    PyConfig_InitPythonConfig(&config);

    // 3. ���þ��Ը���
    config.isolated = 1;       // ��߸��뼶��
    config.use_environment = 0; // ��ȫ���Ի�������
    config.configure_c_stdio = 0; // ��ֹ��׼���ض���

    // 4. ǿ������·��
    std::filesystem::path exe_dir = get_exe_path();
    std::wstring python_home = exe_dir.wstring();
    PyConfig_SetString(&config, &config.home, python_home.c_str());

    // ��ȷ����ģ������·���Ĵ���
    const wchar_t* new_paths[] = {
        (exe_dir / "DLLs").wstring().c_str(),
        (exe_dir / "Lib").wstring().c_str(),
        exe_dir.wstring().c_str()
    };
    Py_ssize_t path_count = sizeof(new_paths) / sizeof(new_paths[0]);

    // ��ȷ���÷�ʽ
    PyStatus status = PyConfig_SetWideStringList(
        &config,
        &config.module_search_paths,
        path_count,  // ·��������Py_ssize_t ���ͣ�
        const_cast<wchar_t**>(new_paths)  // ·������
    );
    
    // 6. ��ʼ�� Python ������
    status = Py_InitializeFromConfig(&config);
    if (PyStatus_Exception(status))
    {
        std::cerr << "Python ��ʼ��ʧ��: " << status.err_msg << std::endl;
        PyConfig_Clear(&config);
        return 1;
    }    

    // 8. ����ģ�鲢���ú���
    PyObject* pModule = PyImport_ImportModule("image2cad");
    if (!pModule)
    {
        PyErr_Print();
        Py_Finalize();
        PyConfig_Clear(&config);
        return 1;
    }

    // ��ȡ png_to_svg ����
    PyObject* pFunc = PyObject_GetAttrString(pModule, "png_to_svg");
    if (!pFunc || !PyCallable_Check(pFunc))
    {
        std::cerr << "�޷��ҵ� png_to_svg �����������ǿɵ��ö���" << std::endl;
        Py_XDECREF(pFunc);
        Py_XDECREF(pModule);
        Py_Finalize();
        PyConfig_Clear(&config);
        return 1;
    }

    // ׼���������ļ�·����
    const char* file_path = R"(D:/Image2CADPy/TestData/pdfImages/page_28_image_1.png)";  // ʹ��ԭʼ�ַ�������ת������
    PyObject* pArgs = PyTuple_New(1);
    PyTuple_SetItem(pArgs, 0, PyUnicode_FromString(file_path));

    // ���� png_to_svg ����
    PyObject* pResult = PyObject_CallObject(pFunc, pArgs);
    if (!pResult)
    {
        std::cerr << "���� png_to_svg ����ʧ�ܣ�" << std::endl;
        PyErr_Print();  // ��ӡ Python ������Ϣ
    }
    else
    {
        std::cout << "png_to_svg �������óɹ���" << std::endl;
        Py_XDECREF(pResult);
    }

    // ����
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
