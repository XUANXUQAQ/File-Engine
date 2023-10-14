﻿/* DO NOT EDIT THIS FILE - it is machine generated */
#include "jni.h"
/* Header for class file_engine_dllInterface_GetHandle */

#ifndef _Included_file_engine_dllInterface_GetHandle
#define _Included_file_engine_dllInterface_GetHandle
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     file_engine_dllInterface_GetHandle
 * Method:    start
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_GetHandle_start
  (JNIEnv *, jobject);

/*
 * Class:     file_engine_dllInterface_GetHandle
 * Method:    stop
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_GetHandle_stop
  (JNIEnv *, jobject);

/*
 * Class:     file_engine_dllInterface_GetHandle
 * Method:    changeToAttach
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_file_engine_dllInterface_GetHandle_changeToAttach
  (JNIEnv *, jobject);

/*
 * Class:     file_engine_dllInterface_GetHandle
 * Method:    changeToNormal
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_file_engine_dllInterface_GetHandle_changeToNormal
  (JNIEnv *, jobject);

/*
 * Class:     file_engine_dllInterface_GetHandle
 * Method:    getExplorerX
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_file_engine_dllInterface_GetHandle_getExplorerX
  (JNIEnv *, jobject);

/*
 * Class:     file_engine_dllInterface_GetHandle
 * Method:    getExplorerY
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_file_engine_dllInterface_GetHandle_getExplorerY
  (JNIEnv *, jobject);

/*
 * Class:     file_engine_dllInterface_GetHandle
 * Method:    getExplorerWidth
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_file_engine_dllInterface_GetHandle_getExplorerWidth
  (JNIEnv *, jobject);

/*
 * Class:     file_engine_dllInterface_GetHandle
 * Method:    getExplorerHeight
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_file_engine_dllInterface_GetHandle_getExplorerHeight
  (JNIEnv *, jobject);

/*
 * Class:     file_engine_dllInterface_GetHandle
 * Method:    getExplorerPath
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_file_engine_dllInterface_GetHandle_getExplorerPath
  (JNIEnv *, jobject);

/*
 * Class:     file_engine_dllInterface_GetHandle
 * Method:    isDialogWindow
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_file_engine_dllInterface_GetHandle_isDialogWindow
  (JNIEnv *, jobject);

/*
 * Class:     file_engine_dllInterface_GetHandle
 * Method:    getToolBarX
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_file_engine_dllInterface_GetHandle_getToolBarX
  (JNIEnv *, jobject);

/*
 * Class:     file_engine_dllInterface_GetHandle
 * Method:    getToolBarY
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_file_engine_dllInterface_GetHandle_getToolBarY
  (JNIEnv *, jobject);

/*
 * Class:     file_engine_dllInterface_GetHandle
 * Method:    isKeyPressed
 * Signature: (I)Z
 */
JNIEXPORT jboolean JNICALL Java_file_engine_dllInterface_GetHandle_isKeyPressed
  (JNIEnv *, jobject, jint);

/*
 * Class:     file_engine_dllInterface_GetHandle
 * Method:    isForegroundFullscreen
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_file_engine_dllInterface_GetHandle_isForegroundFullscreen
  (JNIEnv *, jobject);

/*
 * Class:     file_engine_dllInterface_GetHandle
 * Method:    setEditPath
 * Signature: (Ljava/lang/String;Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_GetHandle_setEditPath
  (JNIEnv *, jobject, jstring, jstring);

#ifdef __cplusplus
}
#endif
#endif
