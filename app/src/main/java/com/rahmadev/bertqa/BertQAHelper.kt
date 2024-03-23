package com.rahmadev.bertqa

import android.content.Context
import android.os.Build
import android.util.Log
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.text.qa.BertQuestionAnswerer
import org.tensorflow.lite.task.text.qa.QaAnswer

class BertQAHelper(
    val context: Context,
    private val resultAnswerListener: ResultAnswerListener?
) {
    private var bertQuestionAnswer: BertQuestionAnswerer? = null

    fun clearBertQuestionAnswerer() {
        bertQuestionAnswer?.close()
        bertQuestionAnswer = null
    }

    private fun setupBertQuestionAnswerer() {
        val baseOptionsBuilder = BaseOptions.builder()

        if (CompatibilityList().isDelegateSupportedOnThisDevice) {
            baseOptionsBuilder.useGpu()
        } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
            baseOptionsBuilder.useNnapi()
        } else {
            baseOptionsBuilder.setNumThreads(4)
        }

        val options = BertQuestionAnswerer.BertQuestionAnswererOptions.builder()
            .setBaseOptions(baseOptionsBuilder.build())
            .build()

        try {
            bertQuestionAnswer = BertQuestionAnswerer.createFromFileAndOptions(context, BERT_QA_MODEL, options)
        } catch (e: Exception) {
            resultAnswerListener?.onError("Bert Question Answerer gagal untuk terinisialisasi")
            Log.e(TAG, "setupBertQuestionAnswerer: " + e.message)
        }
    }

    fun getQuestionAnswer(topicsContent: String, question: String) {
        if (bertQuestionAnswer == null) {
            setupBertQuestionAnswerer()
        }

        var inferenceTime = System.currentTimeMillis()
        val answers = bertQuestionAnswer?.answer(topicsContent, question)
        inferenceTime = System.currentTimeMillis() - inferenceTime

        resultAnswerListener?.onResult(answers, inferenceTime)
    }

    interface ResultAnswerListener {
        fun onError(error: String)
        fun onResult(
            results: List<QaAnswer>?,
            inferenceTimeLong: Long
        )
    }

    companion object {
        private const val BERT_QA_MODEL = "mobilebert.tflite"
        private const val TAG = "BertQaHelper"
    }
}