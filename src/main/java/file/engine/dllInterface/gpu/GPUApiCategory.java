package file.engine.dllInterface.gpu;

/**
 * TODO 添加其他API
 */
enum GPUApiCategory {
    CUDA("cuda");
    final String category;

    GPUApiCategory(String category) {
        this.category = category;
    }

    @Override
    public String toString() {
        return this.category;
    }

    static GPUApiCategory categoryFromString(String c) {
        return switch (c) {
            case "cuda" -> CUDA;
            default -> null;
        };
    }
}
