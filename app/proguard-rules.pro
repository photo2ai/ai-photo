-dontwarn org.conscrypt.**
-dontwarn org.bouncycastle.**
-dontwarn org.openjsse.**

# ML Kit：组件注册类是按 AndroidManifest 里登记的类名反射创建的，R8 看不到这层调用，会把它们（或无参构造）优化掉，
# 运行时取到 null（release 包人像抠图报 NullPointerException）。整包保留，只多几百 KB 的 dex
-keep class * implements com.google.firebase.components.ComponentRegistrar { <init>(); }
-keep class com.google.mlkit.** { *; }
-keep class com.google.android.gms.internal.mlkit_** { *; }
-keep class com.google.firebase.components.** { *; }

# ONNX Runtime：原生库通过 JNI 按类名、方法名回调 Java 类（OnnxTensor、OrtSession 等），混淆或删掉就会崩
-keep class ai.onnxruntime.** { *; }
