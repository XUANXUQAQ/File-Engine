package file.engine.utils.clazz.scan;

import java.io.File;
import java.util.HashSet;
import java.util.Set;

public class FileScanner implements Scan {

    private String defaultClassPath;

    public void setDefaultClassPath(String defaultClassPath) {
        this.defaultClassPath = defaultClassPath;
    }

    private static class ClassSearcher {
        private final Set<String> classPaths = new HashSet<>();

        private Set<String> doPath(File file, String packageName, boolean flag) {

            if (file.isDirectory()) {
                //文件夹我们就递归
                File[] files = file.listFiles();
                if (!flag) {
                    packageName = packageName + "." + file.getName();
                }
                if (files != null) {
                    for (File f1 : files) {
                        doPath(f1, packageName, false);
                    }
                }
            } else {
                //标准文件
                //标准文件我们就判断是否是class文件
                if (file.getName().endsWith(CLASS_SUFFIX)) {
                    //如果是class文件我们就放入我们的集合中。
                    classPaths.add(packageName + "." + file.getName().substring(0, file.getName().lastIndexOf(".")));
                }
            }
            return classPaths;
        }
    }

    @Override
    public Set<String> search(String packageName) {
        //先把包名转换为路径,首先得到项目的classpath
        String classpath = defaultClassPath;
        //然后把我们的包名basPack转换为路径名
        String basePackPath = packageName.replace(".", File.separator);
        String searchPath = classpath + basePackPath;
        return new ClassSearcher().doPath(new File(searchPath), packageName, true);
    }
}

