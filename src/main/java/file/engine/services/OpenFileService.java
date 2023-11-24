package file.engine.services;

import file.engine.annotation.EventRegister;
import file.engine.event.handler.Event;
import file.engine.event.handler.impl.open.file.OpenFileEvent;
import file.engine.services.utils.OpenFileUtil;
import file.engine.utils.file.FileUtil;
import lombok.extern.slf4j.Slf4j;

import java.io.IOException;

@Slf4j
public class OpenFileService {

    @EventRegister(registerClass = OpenFileEvent.class)
    private static void dispatcher(Event event) {
        OpenFileEvent openFileEvent = (OpenFileEvent) event;
        OpenFileEvent.OpenStatus openStatus = openFileEvent.openStatus;
        switch (openStatus) {
            case NORMAL_OPEN -> openFile(openFileEvent.path);
            case LAST_DIR -> openParentPath(openFileEvent.path);
            case WITH_ADMIN -> openFileWithAdmin(openFileEvent.path);
            default -> throw new RuntimeException("error open status");
        }
    }

    private static void openFile(String path) {
        OpenFileUtil.openWithoutAdmin(path);
    }

    private static void openFileWithAdmin(String path) {
        OpenFileUtil.openWithAdmin(path);
    }

    private static void openParentPath(String path) {
        try {
            OpenFileUtil.openFolderByExplorer(path);
        } catch (IOException e) {
            log.error("error: {}", e.getMessage(), e);
            String parentPath = FileUtil.getParentPath(path);
            openFile(parentPath);
        }
    }
}
