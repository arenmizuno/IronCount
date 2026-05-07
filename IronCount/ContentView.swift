//
//  ContentView.swift
//  IronCount
//
//  Created by Aren Mizuno on 5/4/26.
//

import SwiftUI
import Combine

struct ContentView: View {

    @StateObject var camera = CameraManager()

    var body: some View {
        ZStack {
            CameraPreview(session: camera.session)
                .ignoresSafeArea()

            VStack {
                Text(camera.exerciseLabel)
                    .font(.largeTitle)
                    .foregroundColor(.white)

                Text("Reps: \(camera.reps)")
                    .font(.title)
                    .foregroundColor(.green)

                Spacer()
            }
            .padding()
        }
        .onAppear {
            camera.start()
        }
    }
}
