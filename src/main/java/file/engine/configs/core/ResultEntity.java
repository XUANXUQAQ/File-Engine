package file.engine.configs.core;

import java.util.List;

public record ResultEntity(String uuid, List<String> data, int nextIndex, boolean isDone) {
}
