#!/bin/bash

# 配置MSYS2字体环境自动化脚本
# 需要管理员权限执行

# 定义关键路径
MSYS2_PREFIX="/e/msys64"  # 根据实际安装路径修改
WIN_FONTS="/c/Windows/Fonts"
MSYS2_FONT_DIR="${MSYS2_PREFIX}/mingw64/share/fonts"
FONT_CONFIG_DIR="${MSYS2_PREFIX}/mingw64/etc/fonts"

# 1. 创建字体符号链接
echo "🔍 检查并创建字体目录..."

# 确保字体目录存在
mkdir -p "${MSYS2_FONT_DIR}" || {
    echo "❌ 无法创建字体目录：${MSYS2_FONT_DIR}"
    exit 1
}

# 创建符号链接（带路径验证）
if [ -d "${WIN_FONTS}" ]; then
    echo "🔗 创建Windows字体目录符号链接..."
    ln -sfv "${WIN_FONTS}" "${MSYS2_FONT_DIR}/windows_fonts" || {
        echo "❌ 符号链接创建失败，请检查："
        echo "   - 源目录是否存在: ls -l ${WIN_FONTS}"
        echo "   - 目标路径权限: ls -ld ${MSYS2_FONT_DIR}"
        exit 2
    }
else
    echo "⚠️ 未找到Windows字体目录：${WIN_FONTS}"
    echo "请手动确认以下路径是否存在："
    echo "1. 打开文件资源管理器"
    echo "2. 访问路径: C:\Windows\Fonts"
    exit 3
fi

# 2. 生成字体配置文件
echo "📝 生成字体配置文件..."
mkdir -p "${FONT_CONFIG_DIR}/conf.d"

cat > "${FONT_CONFIG_DIR}/fonts.conf" <<EOF
<?xml version="1.0"?>
<!DOCTYPE fontconfig SYSTEM "fonts.dtd">
<fontconfig>
  <!-- 添加Windows字体目录 -->
  <dir>${WIN_FONTS}</dir>
  
  <!-- MSYS2系统字体目录 -->
  <dir>${MSYS2_FONT_DIR}</dir>
  
  <!-- 缓存目录 -->
  <cachedir>${MSYS2_PREFIX}/var/cache/fontconfig</cachedir>
  
  <config>
    <!-- 自动扫描子目录 -->
    <rescan>
      <int>30</int>
    </rescan>
  </config>
</fontconfig>
EOF

# 3. 更新字体缓存
echo "🔄 更新字体缓存..."
fc-cache -fv || {
    echo "❌ 字体缓存更新失败"
    exit 2
}

# 4. 验证配置
echo "✅ 验证字体配置..."
echo "已识别的中文字体："
fc-list : family | grep -i "song|hei|kai|fang" | sort | uniq

# 5. 设置环境变量（可选）
echo "💡 建议将以下内容添加到 shell 配置文件中："
echo "------------------------------------------------"
echo "export FONTCONFIG_PATH='${FONT_CONFIG_DIR}'"
echo "export FONTCONFIG_FILE='fonts.conf'"
echo "------------------------------------------------"

echo "🎉 字体配置完成！现在可以正常使用text2image工具"